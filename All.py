import numpy as np
import matplotlib.pyplot as plt
import math

###########entering raw data


ADD = np.array([1	,1,	1,	1,	1	,1,	1	,1,	1	,1	,1,	1	,1	,1	,1,	1])
AgeGroups_1_0  = np.array([0.195844156,	0.953333333,	0.813333333	,0.70628571	,0.538333333,	0.28	,0,	0.8	,1	,0.38,	0.151168831,0.946666667,	0.786666667,	0.704571429	,0.368333333	,0.19])
AgeGroups_2_0 = np.array([0.198961039	,0.966666667	,0.76,	0.70171429,	0.565	,0.52	,0,	0.18,	1	,0.02,	0.156883117	,0.886666667,	0.73	,0.688571429,	0.393333333	,0.32])
AnyAllr = np.array([0.925714286,	0.96,	1	,0.88971429	,0.951666667,	0.96,	0	,0.9,	0.04,	0	,0.918701299,	0.286666667	,0.95,	0.848571429,	0.863333333,	0.79])
AnyPars3 = np.array([0.930909091,	0.96,	0.983333333	,0.89257143,	0.87,	0.91,	0	,1,	0	,0.16,	0.929090909	,0.893333333,	0.95,	0.825714286	,0.863333333,	0.74])
CookArea = np.array([0.90987013	,0.973333333,	0.896666667	,0.804	,0.863333333	,0.72,	1,	1,	1	,0.36	,0.885974026,	0.78	,0.973333333	,0.828571429	,0.79,	0.69])
DEWORM  = np.array([0.445974026	,1,	0.993333333	,0.99542857	,0.99,	0.99,	0	,1	,1,	0.18,	0.802597403	,0.96,	0.996666667	,0.822857143,	0.828333333	,0.77])
FamilyGrouped_1_0 = np.array([0.219480519	,0.873333333	,0.783333333	,0.69314286,	0.533333333	,0.75,	0	,0.84	,0	,0.4	,0.165714286,	0.8	,0.773333333	,0.757714286	,0.493333333,	0.58])
FamilyGrouped_2_0 = np.array([0.318961039	,0.986666667,	0.513333333	,0.63485714	,0.6,	0.79,	0,	0.58,	0.94	,0.24	,0.220779221	,0.893333333,	0.986666667,	0.84,	0.68,	0.61 ])
GCAT_1  = np.array([0.43038961,	1	,0.906666667	,0.96057143	,0.96	,0.85,	0,	1	,0.7,	0.9	,0.788831169	,0.773333333	,0.946666667	,0.770285714,	0.683333333	,0.37])
GCAT_2 = np.array([0.415584416,	0.993333333	,0.833333333,	0.94285714,	0.875	,0.81	,1	,1,	1,	1	,0.815844156,	0.866666667	,0.57,	0.805714286	,0.433333333,	0.28])
GCOW = np.array([0.800519481,	1	,0.84	,0.83942857,	0.808333333	,0.16,	0,	1	,0.54,	0.06	,0.652727273	,0.753333333	,0.663333333	,0.724571429	,0.426666667,	0.11])
GDOG_1_0  = np.array([0.90987013,	0.106666667,	0.936666667,	0.85714286,	0.943333333	,0.7,	1	,0.26,	1	,0,	0.853766234,	0.9	,0.233333333,	0.34,	0.508333333	,0.46])
GDOG_2_0 = np.array([0.303116883,	0.966666667	,0.79	,0.848,	0.835	,0.47,	0	,0.44,	0	,0.16	,0.161038961,	0.886666667	,0.836666667	,0.725142857	,0.383333333,	0.2])
GELEC_1_0  = np.array([0.681558442,	0.893333333,	0.676666667	,0.71257143	,0.561666667,	0.19,	1	,0.96	,1,	0.3	,0.45038961	,0.973333333,	0.973333333	,0.988,	0.915,	0.9])
GELEC_2_0  = np.array([0.814545455,	0.893333333	,0.536666667	,0.74628571,	0.531666667	,0.45	,0,	1,	1	,1	,0.455324675	,1,	1	,1,	1	,1])
GFLOOR6A_1_0 = np.array([0.217922078	,0.96,	0.9	,0.768,	0.631666667,	0.62,	1,	0	,1,	0,	0.278961039	,0.913333333,	0.14,0.359428571	,0.446666667,	0.39])
GFLOOR6A_2_0 = np.array([0.232207792	,0.973333333,	0.813333333	,0.77657143,	0.61	,0.61,	0	,0.42,	0	,0.12	,0.196363636	,0.926666667,	0.776666667	,0.724	,0.388333333	,0.87])
GFLOOR6A_9_0 = np.array([0.236883117	,0.74,	0.963333333	,0.80342857,	0.638333333	,0.65	,1	,0,	1	,0,	0.260779221	,0.946666667	,0.396666667,	0.469142857,	0.408333333	,0.68])
GWASTE_1_0  = np.array([0.232727273	,0.933333333	,0.833333333	,0.78285714	,0.503333333	,0.07	,0.54	,0.3,	1	,0.02	,0.236103896,	0.553333333	,0.97	,0.804571429	,0.543333333	,0.65])
GWASTE_2_0  = np.array([0.908051948,	0.993333333	,0.276666667,	0.41371429,	0.668333333	,0.42	,0,	1,	0.14,	1	,0.418441558	,0.96,	0.87,	0.929142857,	0.916666667,	0.47])
GWASTE_3_0  = np.array([0.205974026	,0.96	,0.853333333	,0.71257143,	0.521666667	,0.28,	0	,0.88	,0.64	,0.4	,0.182857143	,0.706666667	,0.833333333	,0.750857143,	0.38,	0])
HCIGR6A = np.array([0.825714286	,0.94,	0.953333333,	0.81085714,	0.755	,0.5,	1	,0.96,	1	,0.04,	0.930649351	,0.986666667,	0.92,	0.98,	0.911666667	,0.69])
ToiletType_1_0 = np.array([0.238701299	,0.973333333	,0.493333333	,0.77371429	,0.473333333	,0.38,	0	,1,	1	,1	,0.414805195	,0.966666667	,0.913333333,	0.98,	0.876666667,	0.76])
ToiletType_2_0 = np.array([0.327272727	,0.96,	0.256666667,	0.56057143,	0.575	,0.62,	0.46,	1	,1,	0.9	,0.170649351,	0.906666667	,0.033333333,	0.290285714,	0.206666667,	0.29])
WaterSource_1_0 = np.array([0.327012987	,0.98,	0.22	,0.45885714	,0.606666667,	0.36	,0	,0.94,	1	,0.36	,0.152987013	,0.833333333	,0.476666667	,0.758857143,	0.298333333	,0.19])
WaterSource_2_0 = np.array([0.232727273,	0.966666667,	0.39,	0.56171429,	0.36,	0.24	,1,	0.54	,1,	0	,0.136883117,	0.846666667,	0.076666667	,0.325142857,	0.151666667	,0.26])




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

ADD_std = np.std(ADD)/math.sqrt(16)
AgeGroups_1_0_std=np.std(AgeGroups_1_0)/math.sqrt(16)
AgeGroups_2_0_std=np.std(AgeGroups_2_0)/math.sqrt(16)
AnyAllr_std=np.std(AnyAllr)/math.sqrt(16)
AnyPars3_std=np.std(AnyPars3)/math.sqrt(16)
CookArea_std=np.std(CookArea)/math.sqrt(16)
DEWORM_std=np.std(DEWORM)/math.sqrt(16)
FamilyGrouped_1_0_std=np.std(FamilyGrouped_1_0)/math.sqrt(16)
FamilyGrouped_2_0_std=np.std(FamilyGrouped_2_0)/math.sqrt(16)
GCAT_1_std=np.std(GCAT_1)/math.sqrt(16)
GCAT_2_std=np.std(GCAT_2)/math.sqrt(16)
GCOW_std=np.std(GCOW)/math.sqrt(16)
GDOG_1_0_std=np.std(GDOG_1_0)/math.sqrt(16)
GDOG_2_0_std=np.std(GDOG_2_0)/math.sqrt(16)
GELEC_1_0_std=np.std(GELEC_1_0)/math.sqrt(16)
GELEC_2_0_std=np.std(GELEC_2_0)/math.sqrt(16)
GFLOOR6A_1_0_std=np.std(GFLOOR6A_1_0)/math.sqrt(16)
GFLOOR6A_2_0_std=np.std(GFLOOR6A_2_0)/math.sqrt(16)
GFLOOR6A_9_0_std=np.std(GFLOOR6A_9_0)/math.sqrt(16)
GWASTE_1_0_std=np.std(GWASTE_1_0)/math.sqrt(16)
GWASTE_2_0_std=np.std(GWASTE_2_0)/math.sqrt(16)
GWASTE_3_0_std=np.std(GWASTE_3_0)/math.sqrt(16)
HCIGR6A_std=np.std(HCIGR6A)/math.sqrt(16)
ToiletType_1_0_std=np.std(ToiletType_1_0)/math.sqrt(16)
ToiletType_2_0_std=np.std(ToiletType_2_0)/math.sqrt(16)
WaterSource_1_0_std=np.std(WaterSource_1_0)/math.sqrt(16)
WaterSource_2_0_std=np.std(WaterSource_2_0)/math.sqrt(16)




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
ax.set_title('All Based',fontsize =30)
ax.yaxis.grid(True)

# Save the figure and show
plt.xticks(rotation =90)
plt.yticks(np.arange(0,1.5,step=0.5),fontsize =15)#
plt.tight_layout()
plt.savefig('all_based_plot_with_error_bars.png')
plt.show()





######## save the figure and show
