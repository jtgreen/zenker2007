from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math

class experiment:
    def __init__(self):

        # Experiment 
        # fluidChangeRes

        fluidVolume = 1000 # % resuscitation volume

        self.startTime = 600
        self.endTime = 1600
        self.startTimeRes = 2000
        self.endTimeRes = 3000
        self.totalVolume = fluidVolume

        # total experiment time
        self.tspan = [0, 3600] # % 60 minutes

        # resolution
        self.resolution = 10000

        # %% parameters, finalized for paper...
        self.apSetpoint = 70 # % setpoint for baroreflex regulation of mean arterial pressure; % setpoint for baroreflex regulation of mean arterial pressure
        self.satSna = 25 # % deviation from setpoint in mmHg where SNA reaches 99% saturation either way; % deviation from setpoint in mmHg where SNA reaches 99% saturation either way
        self.Ca = 4 # % arterial compliance, Ursino 1994; % arterial compliance
        self.VaUnstressed = 700 # % unstressed arterial volume; % venous compliance
        self.Va_relaxed = self.VaUnstressed # % unstressed arterial volume
        self.minVvUnstressed = 2700
        self.maxVvUnstressed = 3100
        self.minVv_relaxed = self.minVvUnstressed # % and venous (controlled)
        self.maxVv_relaxed = self.maxVvUnstressed
        self.Cv = 111.11 # % Ursino 1994

        self.hrMin = 40/60 # % minimum heart rate
        self.hrMax = 180/60 # % maximum heart rate

        self.Tsyst = 1/self.hrMax * 0.8 # % duration of systole, constant for now such that systole takes 80% of cycle at max heart rate, Friedberg 2006 fig. 4A; % duration of systole, constant for now
        self.RartMin = 0.5*1.067 # % minimal arterial resistance (at 0 SNA); %relative to Ursino, 1994; % minimal arterial resistance (at 0 SNA)
        self.RartMax = 2*1.067 # % maximal arterial resistance (at 1 SNA); % maximal arterial resistance (at 1 SNA)
        self.Rvalve = 2.5E-3 # % resistance "tricuspid" valve, from Ursino, 1998; % resistance "tricuspid" valve                                 

        # % use fitting results for diastolic compliance pars
        fitResults = scipy.io.loadmat('zenker_quantdiag_code/opt_p0lv_kelv_v0lv_glower_control.mat')
        self.P0lv = fitResults['P0lv'][0][0]
        self.kelv = fitResults['kelv'][0][0] # % ventricular relaxation constant; % scaling constant for ventricular relaxation
        self.Vlved0 = fitResults['V0lv'][0][0] # % unstressed volume LV

        self.cfactor = 1 # % multiply contractility response range by this...

        self.contractilityMin  = self.cfactor * 0.5 * 69 / 1.33 # % half of mean from global study in Glower, 1985
        self.contractilityMax = self.cfactor * 2 * 69/1.33 # % and twice the mean
        self.lowPassCutoff = 0.008 # % cutoff of peripheral effector lowpass, using time constant for unstressed venous volume from Ursino, 1998

        self.fixedSNA = -1 # % closed loop; % if < 0, baroreflex feedback loop closed, otherwise SNA fixed to that level

        # %% end model parameters

        # %% initial conditions etc.
        totalVolume = 4825

        # % split by unstressed volumes for intial guess
        meanVvU = (self.minVvUnstressed + self.maxVvUnstressed)/2
        Va = totalVolume/ (self.VaUnstressed + meanVvU) * self.VaUnstressed
        Vv = totalVolume/ (self.VaUnstressed + meanVvU) * meanVvU
        self.y0 = [Va, Vv, 0.5, 3 * self.Vlved0, 2*self.Vlved0] # % start with midpoint sna, really rough guesses for Vlvs

    def Pa(self, Va, Ca, Va_relaxed):
        # % calculate arterial pressure
        return (Va - Va_relaxed) / Ca #pressure

    def Pv(self, Vv, Cv, Vv_relaxed):
        return (Vv - Vv_relaxed) / Cv #pressure

    def PvComplete(self, Vv, Cv, minVv_relaxed, maxVv_relaxed, snaEffector):
        return (Vv - ((maxVv_relaxed - minVv_relaxed) * (1 - snaEffector) + minVv_relaxed)) / Cv # pressure

    def calcSna(self, delayedAp, apSetpoint, satSna):
        # % calculate sympathetic nervous activity from arterial pressure, satSna
        # % determines 99% saturation point
        return 1 - ( 1 / (1 + np.exp(-4.595119850 * (delayedAp - apSetpoint) / satSna)) ) #sna

    def vlves(self, C, Vlved, Vlved0, Pej, P0lv, kelv):
        # """calculate endsystolic volume as a function of a lot of other things...

        denom = Pej-P0lv*(np.exp(kelv*(Vlved-Vlved0))-1) # % pressure difference between large arteries and ventricle at end-diastole

        # if denom > 0: # art. pressure > Plved, work to perform
        #     # JTG: max([a b])    concatenate((a,b)).max()    max of all values in two vectors
        #     #return np.max([Vlved0, Vlved-C*(Vlved-Vlved0)/denom]) # Vlves

        #     return np.concatenate(([Vlved0], [Vlved-C*(Vlved-Vlved0)/denom])).max()
        # else: # no work, maximally contract
        #     return Vlved0 # Vlves

        _ = np.where(denom>0, Vlved-C*(Vlved-Vlved0)/denom, Vlved0)
        return np.where(denom>_, Vlved0, _)

    def vlved(self, t_ed, Pcvp, kelv, P0lv, Rvalve, Ves, Vlved0):
        # """calculate enddiastolic volume as a function of a lot of other things...

        # JTG: essentially test if blood coming from central venous pressure is greater mmHg than ventricle at end of systole -- P_CVP > P_ES, ref eq.'s 39 + 29 Curcio
        deltap = Pcvp + P0lv * (1 - np.exp(kelv * (Ves - Vlved0))) # % pressure difference between central veins and ventricle drives filling

        # JTG: Old logic doesn't translate well --
        # if deltap > 0:
        #     k1 = -P0lv / Rvalve * np.exp(-kelv * Vlved0)
        #     k2 = kelv
        #     k3 = (Pcvp + P0lv)/Rvalve
            
        #     # JTG: may need alot of floating point work here
        #     return -1 / k2 * np.log(k1 / k3 * (np.exp(-k2 * k3 * t_ed)-1)+np.exp(-k2*(Ves+k3 * t_ed))) # % solution of ODE, see paper; Vlved
        # else:
        #     return Ves # % can't fill => ventricular volume remains at current endsystolic volume 

        return np.where(deltap>0, ( -1 / kelv * np.log( (-P0lv / Rvalve * np.exp(-kelv * Vlved0)) / ((Pcvp + P0lv)/Rvalve) * (np.exp(-kelv * ((Pcvp + P0lv)/Rvalve) * t_ed)-1)+np.exp(-kelv*(Ves+((Pcvp + P0lv)/Rvalve) * t_ed))) ), Ves)

    def volumeChange(self, t, startTime, endTime, startTimeRes, endTimeRes, totalVolume):
        # % input function for model to simulate fluid resuscitation/withdrawal
        if (t >= startTime and t < endTime):
            return -totalVolume/(endTime-startTime) # dVdt
        elif (t >= startTimeRes and t < endTimeRes):
            return totalVolume/(endTimeRes-startTimeRes) # dVdt
        else:
            return 0 # dVdt

    # % primitive cardiovascular model for proof of concept of differential
    # % diagnosis/deterministic model approach
    def odes(self, y, t):

        # % states
        Va = y[0] # % volume in arterial compartment
        PaDel = self.Pa(y[0], self.Ca, self.Va_relaxed) # % arterial pressure, delayed by sna delay (from delayed Va)
        Vv = y[1] # % volume in venous compartment

        if self.fixedSNA < 0: # % closed loop
            snaEffector = y[2] # % low-pass filtered peripheral response to SNA
        else:
            # %%%%%%% open reflex
            snaEffector = self.fixedSNA

        # %%% our ventricular filling states
        Vlved = y[3] # % enddiastolic volume
        Vlves = y[4] # % endsystolic volume, difference gives stroke volume...

        # % baroreflex nonlinearity
        sna = self.calcSna(PaDel, self.apSetpoint, self.satSna);
        # % simple low-pass filter
        if self.fixedSNA < 0: # % loop closed
            dSnaEffector = 2 * math.pi * self.lowPassCutoff * (sna - snaEffector)  # % first order low-pass
        # %%% open reflex
        else:
            dSnaEffector = 0

        hr = (self.hrMax-self.hrMin)*snaEffector + self.hrMin

        Rart = (self.RartMax - self.RartMin)*snaEffector + self.RartMin
        C = (self.contractilityMax - self.contractilityMin)*snaEffector + self.contractilityMin

        Vv_relaxed = (self.maxVv_relaxed - self.minVv_relaxed) * (1 - snaEffector) + self.minVv_relaxed

        Pa1 = self.Pa(Va, self.Ca, self.Va_relaxed)
        Pv1 = self.Pv(Vv, self.Cv, Vv_relaxed)

        Rtpr = Rart # % extremely primitive, no venoles or anything
        Ic = (Pa1 - Pv1) / Rtpr # % total flow through capillary streambed

        t_ed = 1 / hr - self.Tsyst # % duration of diastole, endpoint of filling

        Ico = (Vlved - Vlves) * hr # % cardiac output is stroke volume * heart rate

        # dVa
        dVa = Ico - Ic

        # % now calculate next step of discrete iteration at current point
        nextVlves = self.vlves(C, Vlved, self.Vlved0, Pa1, self.P0lv, self.kelv)
        nextVlved = self.vlved(t_ed, Pv1, self.kelv, self.P0lv, self.Rvalve, Vlves, self.Vlved0)

        # dVlves and dVlved
        # % rates of change are average rates of change of discrete dynamical system over a whole beat in current situation
        dVlves = (nextVlves - Vlves)*hr
        dVlved = (nextVlved - Vlved)*hr

        # dVv
        dVv =  -dVa + self.volumeChange(t, self.startTime, self.endTime, 
                    self.startTimeRes, self.endTimeRes, self.totalVolume) #% external volume substitution/withdrawal affects venous compartment

        return [dVa, dVv, dSnaEffector, dVlved, dVlves]

    # plot
    def plot(self):
        # declare a time vector (time window)
        t = np.linspace(self.tspan[0], self.tspan[1], self.resolution)
        y = odeint(self.odes, self.y0, t)

        dVa = y[:,0]
        dVv = y[:,1]
        dSnaEffector = y[:,2]
        dVlved = y[:,3]
        dVlves = y[:,4]

        # Plot results
        ap = self.Pa(dVa, self.Ca, self.VaUnstressed)
        hr = (self.hrMax-self.hrMin)*dSnaEffector + self.hrMin # % heart rate depends linearly on effector SNA
        co = hr * (dVlved - dVlves) # % cardiac output in ml/min = heart rate in bpm *(EDV-ESV)
        VlvesCalc = self.vlves(self.cfactor*((self.contractilityMax - self.contractilityMin)*dSnaEffector + self.contractilityMin), 
            dVlved, self.Vlved0, self.Pa(dVa, self.Ca, self.VaUnstressed), self.P0lv, self.kelv)
        VlvedCalc = self.vlved(
            1/((self.hrMax-self.hrMin)*dSnaEffector + self.hrMin)-self.Tsyst, # % t_ed
            self.Pv(dVv, self.Cv, (self.maxVvUnstressed - self.minVvUnstressed) * (1 - dSnaEffector) + self.minVvUnstressed), # % Pcvp
            self.kelv, self.P0lv, self.Rvalve, dVlves, self.Vlved0)
        sv = dVlved - dVlves # % stroke volume from system states
        svCalc = (VlvedCalc - VlvesCalc) # % and next step of discrete time system

        # Fig 1
        plt.plot(t, ap, 'k-', LineWidth=2)
        plt.plot(t, self.PvComplete(dVv, self.Cv, self.minVvUnstressed, self.maxVvUnstressed, dSnaEffector), 'k--', LineWidth=2)
        plt.plot(t, hr*60, 'k:', LineWidth=2)
        plt.plot(t, co, 'k-.', LineWidth=2)
        plt.legend(['Arterial pressure [mmHg]', 'Venous pressure [mmHg]', 'Heart rate [bpm]', 'Cardiac output [ml/s]'], loc='upper right')

        # disp(sprintf('pre-res AP %d, post-res AP %d', ap(max(find(sol.x < 240))), ap(length(ap))));

        plt.vlines(600, 0, 180, [0.5, 0.5, 0.5])
        plt.vlines(1600, 0, 180, [0.5, 0.5, 0.5])
        plt.vlines(2000, 0, 180, [0.5, 0.5, 0.5])
        plt.vlines(3000, 0, 180, [0.5, 0.5, 0.5])
        plt.annotate(
            'Volume withdrawal 1 ml/s', xy=(1100, 160),xytext=(1100, 160) ,
            horizontalalignment="center",
            arrowprops=dict(arrowstyle='->',lw=1)
        )
        plt.annotate(
            'Volume infusion 1 ml/s', xy=(2500, 160),xytext=(2500, 160) ,
            horizontalalignment="center",
            arrowprops=dict(arrowstyle='->',lw=1)
        )
        plt.show()

        # Fig 2
        plt.plot(t, sv, 'k', LineWidth=2)
        plt.plot(t, svCalc, 'k--', LineWidth=2)

        plt.xlim((0,45))
        plt.legend(['Stroke volume from system states [ml]', 'Stroke volume from next step of discrete time system calculated from current system states [ml]'], loc='lower right');
        plt.ylabel('Stroke volume [mL]')
        plt.xlabel('Time [s]')
        plt.show()

