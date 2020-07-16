import numpy as np
import matplotlib.pyplot as plt
import math

###########entering raw data


ADD = np.array([0])
AgeGroups_1_0  = np.array([1])
AgeGroups_2_0 = np.array([0.3])
AnyAllr = np.array([1])
AnyPars3 = np.array([0])
CookArea = np.array([1])
DEWORM  = np.array([1])
FamilyGrouped_1_0 = np.array([1])
FamilyGrouped_2_0 = np.array([1])
GCAT_1  = np.array([0])
GCAT_2 = np.array([0.98])
GCOW = np.array([0])
GDOG_1_0  = np.array([0])
GDOG_2_0 = np.array([0])
GELEC_1_0  = np.array([0])
GELEC_2_0  = np.array([0])
GFLOOR6A_1_0 = np.array([0])
GFLOOR6A_2_0 = np.array([0.9])
GFLOOR6A_9_0 = np.array([0.04])
GWASTE_1_0  = np.array([1])
GWASTE_2_0  = np.array([0])
GWASTE_3_0  = np.array([1])
HCIGR6A = np.array([1])
ToiletType_1_0 = np.array([1])
ToiletType_2_0 = np.array([0])
WaterSource_1_0 = np.array([0])
WaterSource_2_0 = np.array([0])




######## calculate the average
ADD_mean=np.mean(ADD)
AgeGroups_1_0_mean=np.mean(AgeGroups_1_0)
AgeGroups_2_0_mean=np.mean(AgeGroups_2_0)
AnyAllr_mean =np.mean(AnyAllr)
AnyPars3_mean=np.mean(AnyPars3)
CookArea_mean=np.mean(CookArea)
DEWORM_mean=np.mean(DEWORM)
FamilyGrouped_1_0_mean=np.mean(FamilyGrouped_1_0)
FamilyGrouped_2_0_mean=np.mean(FamilyGrouped_2_0)
GCAT_1_mean=np.mean(GCAT_1)
GCAT_2_mean=np.mean(GCAT_2)
GCOW_mean =np.mean(GCOW)
GDOG_1_0_mean=np.mean(GDOG_1_0)
GDOG_2_0_mean=np.mean(GDOG_2_0)
GELEC_1_0_mean=np.mean(GELEC_1_0)
GELEC_2_0_mean=np.mean(GELEC_2_0)
GFLOOR6A_1_0_mean=np.mean(GFLOOR6A_1_0)
GFLOOR6A_2_0_mean=np.mean(GFLOOR6A_2_0)
GFLOOR6A_9_0_mean=np.mean(GFLOOR6A_9_0)
GWASTE_1_0_mean=np.mean(GWASTE_1_0)
GWASTE_2_0_mean=np.mean(GWASTE_2_0)
GWASTE_3_0_mean=np.mean(GWASTE_3_0)
HCIGR6A_mean=np.mean(HCIGR6A)
ToiletType_1_0_mean=np.mean(ToiletType_1_0)
ToiletType_2_0_mean=np.mean(ToiletType_2_0)
WaterSource_1_0_mean=np.mean(WaterSource_1_0)
WaterSource_2_0_mean=np.mean(WaterSource_2_0)





####### calculate the standard deviation

ADD_std = np.std(ADD)
AgeGroups_1_0_std=np.std(AgeGroups_1_0)
AgeGroups_2_0_std=np.std(AgeGroups_2_0)
AnyAllr_std=np.std(AnyAllr)
AnyPars3_std=np.std(AnyPars3)
CookArea_std=np.std(CookArea)
DEWORM_std=np.std(DEWORM)
FamilyGrouped_1_0_std=np.std(FamilyGrouped_1_0)
FamilyGrouped_2_0_std=np.std(FamilyGrouped_2_0)
GCAT_1_std=np.std(GCAT_1)
GCAT_2_std=np.std(GCAT_2)
GCOW_std=np.std(GCOW)
GDOG_1_0_std=np.std(GDOG_1_0)
GDOG_2_0_std=np.std(GDOG_2_0)
GELEC_1_0_std=np.std(GELEC_1_0)
GELEC_2_0_std=np.std(GELEC_2_0)
GFLOOR6A_1_0_std=np.std(GFLOOR6A_1_0)
GFLOOR6A_2_0_std=np.std(GFLOOR6A_2_0)
GFLOOR6A_9_0_std=np.std(GFLOOR6A_9_0)
GWASTE_1_0_std=np.std(GWASTE_1_0)
GWASTE_2_0_std=np.std(GWASTE_2_0)
GWASTE_3_0_std=np.std(GWASTE_3_0)
HCIGR6A_std=np.std(HCIGR6A)
ToiletType_1_0_std=np.std(ToiletType_1_0)
ToiletType_2_0_std=np.std(ToiletType_2_0)
WaterSource_1_0_std=np.std(WaterSource_1_0)
WaterSource_2_0_std=np.std(WaterSource_2_0)




######### create lists for the plot

Features = ['Place of residence','Cook Area','Any smokers in home','Have cat kept outside','Sometimes use electricity','Never use electricity','Deworming status','Have cat live inside','Toilet type is pit','Any parasites found','Any allergic disease','Dispose waste in open field','Have dog live inside','Have cow','Family size >5','Dispose waste in pit','Age 6-10','Floor type not cement, wood, mud','Toilet is open field','Floor type is wood','Dispose waste by burning','Water source is well','Family size from 4-5','Age 11-15','Floor type is mud','Water source is river/rainwater','Have a dog kept outside']


x_pos = np.arange(len(Features))
CTEs = [ADD_mean,CookArea_mean,HCIGR6A_mean,GCAT_2_mean,GELEC_1_0_mean,GELEC_2_0_mean,DEWORM_mean,GCAT_1_mean,ToiletType_1_0_mean,AnyPars3_mean,AnyAllr_mean,GWASTE_2_0_mean,GDOG_1_0_mean,GCOW_mean,FamilyGrouped_2_0_mean,GWASTE_1_0_mean,AgeGroups_1_0_mean,GFLOOR6A_9_0_mean,ToiletType_2_0_mean,GFLOOR6A_1_0_mean,GWASTE_3_0_mean,WaterSource_1_0_mean,FamilyGrouped_1_0_mean,AgeGroups_2_0_mean,GFLOOR6A_2_0_mean,WaterSource_2_0_mean,GDOG_2_0_mean]
error = [ADD_std,CookArea_std,HCIGR6A_std,GCAT_2_std,GELEC_1_0_std,GELEC_2_0_std,DEWORM_std,GCAT_1_std,ToiletType_1_0_std,AnyPars3_std,AnyAllr_std,GWASTE_2_0_std,GDOG_1_0_std,GCOW_std,FamilyGrouped_2_0_std,GWASTE_1_0_std,AgeGroups_1_0_std,GFLOOR6A_9_0_std,ToiletType_2_0_std,GFLOOR6A_1_0_std,GWASTE_3_0_std,WaterSource_1_0_std,FamilyGrouped_1_0_std,AgeGroups_2_0_std,GFLOOR6A_2_0_std,WaterSource_2_0_std,GDOG_2_0_std]

#########build the plot
fig, ax = plt.subplots()
ax.bar(x_pos,CTEs,yerr = error,align = 'center',alpha = 0.5,color = 'grey',ecolor = 'black',error_kw=dict(lw=1, capsize=2, capthick=1),capsize=4)
ax.set_ylabel('Probability', fontsize =15)
ax.set_xticks(x_pos)
ax.set_xticklabels(Features)
ax.set_title('CFS Based',fontsize =30)
ax.yaxis.grid(True)

# Save the figure and show
plt.xticks(rotation =90)
plt.yticks(np.arange(0,1.5,step=0.5),fontsize =15)#
plt.tight_layout()
plt.savefig('CFS_based_plot_with_error_bars.png')
plt.show()





######## save the figure and show
