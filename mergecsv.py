import csv
import pandas as pd
import os
def readCSI(csi_path):
    with open(csi_path, "r", encoding='utf-8-sig') as csvfile:
        csvreader = csv.reader(csvfile)
        data1 = list(csvreader)
    aa = pd.DataFrame(data1)  # 读取CSI数据到aa

    return aa
def mergecsv_withlabel(a,b,i):
    aa = readCSI(a)
    bb = readCSI(b)
    bb.iloc[:,-1]=i
    aa = pd.concat([aa, bb], ignore_index=True)
    aa.to_csv(a, header=None, index=None)

def mergecsv(a,b):
    aa = readCSI(a)
    bb = readCSI(b)
    bb = bb.iloc[:,0:50]
    aa = pd.concat([aa, bb], ignore_index=True)
    aa.to_csv(a, header=None, index=None)
if __name__ == '__main__':
    head="./data/csi_result_2.4m_apartment_c200/"
    #file=["empty","empty","empty","empty","empty","empty","left_arm","right_arm","stand_far_launcher","stand_near_launcher"]
    file = ["empty","stand_far_launcher","stand_near_launcher"]

    output="nearandfar"
    i=1
    if os.path.exists(head + output + ".csv"):
        os.remove(head + output + ".csv")
    os.system(r"echo a 2> {}".format(head + output + ".csv"))
    for ff in file:
        mergecsv(head+output+".csv",head+ff+".csv")
        i+=1