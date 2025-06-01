from logging.config import valid_ident
import numpy as np
import math as math
from sklearn.linear_model import LinearRegression

class SCB_CONVERSION:
    """SCB conversion class."""
    
    def __init__(self, SNR: np.ndarray, AMP: np.ndarray, Consts: np.ndarray, Vbeam: np.ndarray, Temp: np.ndarray, date_time: np.ndarray) -> None:
        self.SNR = SNR
        #input array of SNR values where first index is SNR#, second index is cell #, and third index is sample #
        self.AMP = AMP
        #input array of AMP values where first index is AMP#, second index is cell #, and third index is sample #
        self.Consts = Consts
        '''
        input array of constants in order: 
        frequency[0], EffectiveDiameter[1], beam_orientation[2], slant_angle[3], 
        Blank_distance[4], Cell_size[5], Number_of_cells[6], Beam_Number[7], 
        Moving_avg_span[8], BS_values[9], Intenscale[10], Rmin[11], Rmax[12],
        Mincells[13], MinVbeam[14], Nearfield[15], Remove_min_WCB[16]
        '''
        self.Vbeam = Vbeam
        self.Temp = Temp
        self.date_time = date_time
    
    def compute_R(self) -> np.ndarray:
        
        Cell_size = self.Consts[5]
        Blank_distance = self.Consts[4]
        Number_of_cells = self.Consts[6]
        slant_angle = self.Consts[3]
        
        FirstCell = Blank_distance + Cell_size / 2;
        LastCell = FirstCell + (Number_of_cells - 1) * Cell_size;

        #mid-point cell distance along the beam
        if(Number_of_cells > 1):
            cos_degrees = np.cos(np.radians(slant_angle))
            coeff = 1/cos_degrees
            R = coeff*np.arange(FirstCell, LastCell + 1, Cell_size)
            R = list(map(lambda el:[el], R))
            R = np.array(R)
        else:
            R = FirstCell;
            
        return R
    
    def compute_MB(self) -> np.ndarray:
        
        BS_values = self.Consts[9]
        Beam_Number = self.Consts[7]
        Intens_scale = self.Consts[10]
        SNR = self.SNR
        AMP = self.AMP
        
        if(BS_values == 'SNR'):
            if(Beam_Number == '1'):
                MB = SNR
            elif(Beam_Number == '2'):
                MB = SNR
            elif(Beam_Number == 'Avg'):
                # do it in one line of code instead of two
                MB = np.mean(SNR, axis = 0)
              
            
        elif(BS_values == 'AMP'):
            if(Beam_Number == '1'):
                MB = Intens_scale*AMP
            elif(Beam_Number == '2'):
                MB = Intens_scale*AMP
            elif(Beam_Number == 'Avg'):
                #do it in one line
                MB = np.mean(SNR, axis = 0)*Intens_scale
               
                    
        return MB
    
    def remove_invalid_cells(self, R, MB) -> None:
        MinVbeam = self.Consts[14]
        Rmin = self.Consts[11]
        Rmax = self.Consts[12]

        #non-loop version of deleting
        V_invalid = self.Vbeam < MinVbeam
        self.Vbeam = np.delete(self.Vbeam, V_invalid, 0)
        MB = np.delete(MB, V_invalid, 1)
        self.Temp = np.delete(self.Temp, V_invalid, 0)
        self.date_time = np.delete(self.date_time, V_invalid, 0)

        R_invalid = ((R < Rmin) |  (R > Rmax)).flatten()
        print(R.shape, 'R_invalid,',R_invalid.shape)
        R = np.delete(R, R_invalid, 0)
        #deleate along axis 1 (go vector by vector)
        MB = np.delete(MB, R_invalid, 0)
        
        j = 0
        while(j < len(self.Vbeam)):
            if(self.Vbeam[j] < MinVbeam):
                self.Vbeam = np.delete(self.Vbeam, j, 0)
                MB = np.delete(MB, j, 1)
                self.Temp = np.delete(self.Temp, j, 0)
                self.date_time = np.delete(self.date_time, j, 0)
            else:
                j = j + 1
    
        i = 0
        while(i < len(R)):
            if(R[i] < Rmin or R[i] > Rmax):
                R = np.delete(R, i, 0)
                MB = np.delete(MB, i, 0)
            else:
                i = i + 1
        return
    
    def compute_WCB(self, R, MB) -> np.ndarray:
        Temp = self.Temp
        frequency = self.Consts[0]
        EffectiveDiameter = self.Consts[1]
        Nearfield = self.Consts[15]

        def speed_of_sound(Temperature):
            c1 = 1.402385*(10**3) + 5.038813*Temperature
            c2 = (5.799136*(10**(-2)))*(Temperature**2)
            c3 = (3.287156*(10**(-4)))*(Temperature**3)
            c4 = (1.398845*(10**(-6)))*(Temperature**4)
            c5 = (2.787860*(10**(-9)))*(Temperature**5)
            return c1-c2+c3-c4+c5

        def range_dependence(R, critical_range):
            return R/critical_range
        
        # getting rid of the loop 
        f_T = 21.9*10**(6-(1520/(Temp+273)))
        alpha_w = 8.686*3.38*(10**-6)*(frequency**2)/f_T
        c = speed_of_sound(Temp)
        lam = c/(frequency*(10**3))
        Rcrit = (math.pi*((EffectiveDiameter/2)**2))/lam
        
        Zz = []
        j = 0
        k = 0
        while(j < len(R)):
            new_list = []
            k = 0
            while(k < len(Rcrit)):
                new_list.append(range_dependence(R[j], Rcrit[k]))
                k = k + 1
            Zz.append(new_list)
            j = j + 1
            
        if(Nearfield):
            Psi = []
            i = 0
            while(i < len(Zz[0])):
                j = 0
                new_list = []
                while(j < len(Zz)):
                    new_list.append((1 + 1.35*Zz[j][i] + (2.5*Zz[j][i])**3.2)/(1.35*Zz[j][i] + (2.5*Zz[j][i])**3.2))
                    j = j + 1
                Psi.append(new_list)
                i = i + 1
        else:
            Psi_vec = np.ones(len(R))
            i = 0
            Psi = []
            while(i < len(Rcrit)):
                Psi.append(Psi_vec)
                i = i + 1
                
        Psi_R = [[] for i in range(len(R))]
        i = 0
        while(i < len(R)):
            j = 0
            while(j < len(Psi)):
                Psi_R[i].append(Psi[j][i] * R[i])
                j = j + 1
            i = i + 1
            
        alpha_w_R = [[] for i in range(len(R))]
        i = 0
        while(i < len(R)):
            j = 0
            while(j < len(alpha_w)):
                alpha_w_R[i].append(alpha_w[j]*R[i])
                j = j + 1
            i = i + 1
            
        if(len(Psi_R) == len(alpha_w_R) and len(Psi_R[1]) == len(alpha_w_R[1])):
            l = 0
            twoTL = [[] for i in range(len(Psi_R))]
            while(l < len(Psi_R)):
                m = 0
                while(m < len(Psi_R[0])):
                    twoTL[l].append(20*np.log10(Psi_R[l][m]) + 2*alpha_w_R[l][m])
                    m = m + 1
                l = l + 1
        else:
            print("Houston we have a problem")
            
        k = 0
        WCB = [[] for i in range(len(twoTL))]
        while(k < len(twoTL)):
            j = 0
            while(j < len(twoTL[0])):
                WCB[k].append(MB[k][j] + twoTL[k][j])
                j = j + 1
            k = k + 1
        
        return WCB
    
    def compute_alpha_s(self, R, WCB) -> np.ndarray:
        
        WCB_T = [[] for i in range(len(WCB[0]))]
        i = 0
        while(i < len(WCB[0])):
            j = 0
            while(j < len(WCB)):
                WCB_T[i].append(WCB[j][i])
                j = j + 1
            i = i + 1

        alpha_s = []
        i = 0
        while(i < len(WCB_T)):
            model = LinearRegression()
            model.fit(R, WCB_T[i])
            alpha_s.append(-0.5*model.coef_)
            i = i + 1
        return alpha_s
    
    def compute_SCB(self, R, WCB, alpha_s) -> np.ndarray:
        
        #for size
        SCB_T = [[] for i in range(len(WCB[0]))]

        i = 0
        while(i < len(WCB[0])):
            j = 0
            while(j < len(WCB)):
                SCB_T[i].append(WCB[j][i] + 2*R[j]*alpha_s[i])
                j = j + 1
            i = i + 1

        SCB = [[] for i in range(len(SCB_T[0]))]
        i = 0
        while(i < len(SCB_T[0])):
            j = 0
            while(j < len(SCB_T)):
                SCB[i].append(SCB_T[j][i])
                j = j + 1
            i = i + 1
            
        return SCB
    
    def compute_Mean_SCB(self, SCB) -> np.ndarray:
        
        Mean_SCB = []
        k = 0
        while(k < len(SCB[0])):
            l = 0
            Mean_one_row = 0
            while(l < len(SCB)):
                Mean_one_row = Mean_one_row + SCB[l][k]
                l = l + 1
            Mean_SCB.append((1/len(SCB))*Mean_one_row)
            k = k + 1
        return Mean_SCB
    
    def convert_SNR_to_Mean_SCB(self) -> np.ndarray:
        
        R = self.compute_R()
        MB = self.compute_MB()
        self.remove_invalid_cells(R, MB)
        WCB = self.compute_WCB(R, MB)
        alpha_s = self.compute_alpha_s(R, WCB)
        SCB = self.compute_SCB(R, WCB, alpha_s)
        Mean_SCB = self.compute_Mean_SCB(SCB)
        #To ensure returns as 2D array
        results = [np.array(Mean_SCB).flatten(), np.array(alpha_s).flatten()]
        return results
        