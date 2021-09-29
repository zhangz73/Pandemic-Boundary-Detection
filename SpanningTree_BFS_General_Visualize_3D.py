import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

dt = pd.read_csv("../Results/BoundaryResultsGeneral_prev.csv")
print("All available durations of the pandemic is: " + str(sorted(list(set(dt["T"])))))
print("Please enter the one(s) you would like to see (empty for all):")
lst = input()
if len(lst) == 0:
    lst = sorted(list(set(dt["T"])))
else:
    lst = sorted(list(set([int(x) for x in lst.split()])))

dt = dt[dt["T"].isin(lst)]

def plot_2D(bar, non_coop_prob, radius, pop, type="People", log_scale=False):
    if bar == 0:
        df = dt[(dt["Bar"] == 0) & (dt["NonCooperativeProb"].round(3) == non_coop_prob) & (dt["Radius"] == radius) & (dt["Population"] == pop)]
    else:
        df = dt[(dt["Bar"] != 0) & (dt["NonCooperativeProb"].round(3) == non_coop_prob) & (dt["Radius"] == radius) & (dt["Population"] == pop)]
    
    df = df[df["InternalPeople"] > 10]
    
    denom_BFS = np.array(df[df.Algorithm == "BFS"][f"ExternalBoundary{type}"]) + np.array(df[df.Algorithm == "BFS"][f"Internal{type}"])
    denom_DFS = np.array(df[df.Algorithm == "DFS"][f"ExternalBoundary{type}"]) + np.array(df[df.Algorithm == "DFS"][f"Internal{type}"])
    denominator = f"{type}InsideExternalBoundary"
    if non_coop_prob == 0:
        FPR_BFS = np.array(df[df.Algorithm == "BFS"][f"FalsePos{type}"]) / denom_BFS
        FPR_DFS = np.array(df[df.Algorithm == "DFS"][f"FalsePos{type}"]) / denom_DFS
    else:
        FPR_BFS = np.array(df[df.Algorithm == "BFS"][f"SusceptibleCoverage{type}"]) / denom_BFS
        FPR_DFS = np.array(df[df.Algorithm == "DFS"][f"SusceptibleCoverage{type}"]) / denom_DFS

    TestRatio_BFS = np.array(df[df.Algorithm == "BFS"][f"TotalTest{type}"]) / denom_BFS
    TestRatio_DFS = np.array(df[df.Algorithm == "DFS"][f"TotalTest{type}"]) / denom_DFS

    #bar = np.array(df.Bar)
    T_BFS = np.array(df[df.Algorithm == "BFS"]["T"])
    p_arr_BFS = np.array(df[df.Algorithm == "BFS"]["p"])
    T_DFS = np.array(df[df.Algorithm == "DFS"]["T"])
    p_arr_DFS = np.array(df[df.Algorithm == "DFS"]["p"])
    p_uniq = sorted(list(set(df["p"])))

    fig = plt.figure(figsize=(12, 8))

    ax = fig.add_subplot(221)#, projection="3d")
    #ax.plot_trisurf(p, T, FNR)
    for t in sorted(list(set(T_BFS))):
        FNR_mu = []
        FNR_95 = []
        FNR_05 = []
        for p in p_uniq:
            FNR_mu.append(np.median(FPR_BFS[(T_BFS == t) & (p_arr_BFS == p)]))
            FNR_95.append(np.percentile(FPR_BFS[(T_BFS == t) & (p_arr_BFS == p)], 95))
            FNR_05.append(np.percentile(FPR_BFS[(T_BFS == t) & (p_arr_BFS == p)], 5))
        FNR_mu = np.array(FNR_mu)
        FNR_95 = np.array(FNR_95)
        FNR_05 = np.array(FNR_05)
        ax.plot(p_uniq, FNR_mu, label=f"Duration of Pandemic = {t}")
        ax.fill_between(p_uniq, FNR_05, FNR_95, alpha=0.1)
    ax.set_xlabel("Infection Probability")
    #ax.set_ylabel("Pandemic Transmission Time")
    ax.set_ylabel("False Positive Rates (Full Search)")
    if log_scale:
        ax.set_yscale("log")
    #ax.set_title(f"False Positive Rates (BFS) = FalsePos / {denominator}")
    ax.legend()

    ax = fig.add_subplot(222)#, projection="3d")
    #ax.plot_trisurf(p, T, FPR)
    for t in sorted(list(set(T_DFS))):
        FPR_mu = []
        FPR_95 = []
        FPR_05 = []
        for p in p_uniq:
            FPR_mu.append(np.median(FPR_DFS[(T_DFS == t) & (p_arr_DFS == p)]))
            FPR_95.append(np.percentile(FPR_DFS[(T_DFS == t) & (p_arr_DFS == p)], 95))
            FPR_05.append(np.percentile(FPR_DFS[(T_DFS == t) & (p_arr_DFS == p)], 5))
        FPR_mu = np.array(FPR_mu)
        FPR_95 = np.array(FPR_95)
        FPR_05 = np.array(FPR_05)
        ax.plot(p_uniq, FPR_mu, label=f"Duration of Pandemic = {t}")
        ax.fill_between(p_uniq, FPR_05, FPR_95, alpha=0.1)
    ax.set_xlabel("Infection Probability")
    #ax.set_ylabel("Pandemic Transmission Time")
    ax.set_ylabel("False Positive Rates (Heuristic Search)")
    if log_scale:
        ax.set_yscale("log")
    #ax.set_title(f"False Positive Rates (DFS) = FalsePos / {denominator}Num")
    ax.legend()

    ax = fig.add_subplot(223)#, projection="3d")
    #ax.plot_trisurf(p, T, TestRatio)
    for t in sorted(list(set(T_BFS))):
        Test_mu = []
        Test_95 = []
        Test_05 = []
        for p in p_uniq:
            Test_mu.append(np.median(TestRatio_BFS[(T_BFS == t) & (p_arr_BFS == p)]))
            Test_95.append(np.percentile(TestRatio_BFS[(T_BFS == t) & (p_arr_BFS == p)], 95))
            Test_05.append(np.percentile(TestRatio_BFS[(T_BFS == t) & (p_arr_BFS == p)], 5))
        Test_mu = np.array(Test_mu)
        Test_95 = np.array(Test_95)
        Test_05 = np.array(Test_05)
        ax.plot(p_uniq, Test_mu, label=f"Duration of Pandemic = {t}")
        ax.fill_between(p_uniq, Test_05, Test_95, alpha=0.1)
    ax.set_xlabel("Infection Probability")
    #ax.set_ylabel("Pandemic Transmission Time")
    ax.set_ylabel(f"# Test Conducted (Full Search) / N")
    if log_scale:
        ax.set_yscale("log")
    #ax.set_title(f"# Test Conducted (BFS) / {denominator}Num")
    ax.legend()
    
    ax = fig.add_subplot(224)#, projection="3d")
    #ax.plot_trisurf(p, T, TestRatio)
    for t in sorted(list(set(T_DFS))):
        Test_mu = []
        Test_95 = []
        Test_05 = []
        for p in p_uniq:
            Test_mu.append(np.median(TestRatio_DFS[(T_DFS == t) & (p_arr_DFS == p)]))
            Test_95.append(np.percentile(TestRatio_DFS[(T_DFS == t) & (p_arr_DFS == p)], 95))
            Test_05.append(np.percentile(TestRatio_DFS[(T_DFS == t) & (p_arr_DFS == p)], 5))
        Test_mu = np.array(Test_mu)
        Test_95 = np.array(Test_95)
        Test_05 = np.array(Test_05)
        ax.plot(p_uniq, Test_mu, label=f"Duration of Pandemic = {t}")
        ax.fill_between(p_uniq, Test_05, Test_95, alpha=0.1)
    ax.set_xlabel("Infection Probability")
    #ax.set_ylabel("Pandemic Transmission Time")
    ax.set_ylabel(f"# Test Conducted (Heuristic Search) / N")
    if log_scale:
        ax.set_yscale("log")
    #ax.set_title(f"# Test Conducted (DFS) / {denominator}Num")
    ax.legend()

#    if non_coop_prob == 0:
#        plt.suptitle("Population = " + str(pop) + " NonCooperativeRate = " + str(non_coop_prob) + "\nType = " + str(type))
#    else:
#        plt.suptitle("Population = " + str(pop) + " NonCooperativeRate = " + str(non_coop_prob) + "\nRadius = " + str(radius) + " Bar = " + str(bar) + "\nType = " + str(type))
    plt.savefig(f"BoundaryGeneralPlots3D/Deliverable/pop={pop}_non-coop-prob={non_coop_prob}_radius={radius}_bar={bar}_log={log_scale}_type={type}.png")
    plt.clf()
    plt.close()

def plot_2D_V2(bar, non_coop_prob, radius, pop, type="People", log_scale=False):
    if bar == 0:
        df = dt[(dt["Bar"] == 0) & (dt["NonCooperativeProb"].round(3) == non_coop_prob) & (dt["Radius"] == radius) & (dt["Population"] == pop)]
    else:
        df = dt[(dt["Bar"] != 0) & (dt["NonCooperativeProb"].round(3) == non_coop_prob) & (dt["Radius"] == radius) & (dt["Population"] == pop)]
    
    df = df[df["InternalPeople"] > 10]
    
    denom_BFS = np.array(df[df.Algorithm == "BFS"][f"ExternalBoundary{type}"]) + np.array(df[df.Algorithm == "BFS"][f"Internal{type}"])
    denom_DFS = np.array(df[df.Algorithm == "DFS"][f"ExternalBoundary{type}"]) + np.array(df[df.Algorithm == "DFS"][f"Internal{type}"])
    denominator = f"{type}InsideExternalBoundary"
    if non_coop_prob == 0:
        FPR_BFS = np.array(df[df.Algorithm == "BFS"][f"FalsePos{type}"]) / denom_BFS
        FPR_DFS = np.array(df[df.Algorithm == "DFS"][f"FalsePos{type}"]) / denom_DFS
    else:
        FPR_BFS = np.array(df[df.Algorithm == "BFS"][f"SusceptibleCoverage{type}"]) / denom_BFS
        FPR_DFS = np.array(df[df.Algorithm == "DFS"][f"SusceptibleCoverage{type}"]) / denom_DFS

    TestRatio_BFS = np.array(df[df.Algorithm == "BFS"][f"TotalTest{type}"]) / denom_BFS
    TestRatio_DFS = np.array(df[df.Algorithm == "DFS"][f"TotalTest{type}"]) / denom_DFS

    #bar = np.array(df.Bar)
    T_BFS = np.array(df[df.Algorithm == "BFS"]["T"])
    p_arr_BFS = np.array(df[df.Algorithm == "BFS"]["p"])
    T_DFS = np.array(df[df.Algorithm == "DFS"]["T"])
    p_arr_DFS = np.array(df[df.Algorithm == "DFS"]["p"])
    p_uniq = sorted(list(set(df["p"])))

    fig, ax = plt.subplots()
    for t in sorted(list(set(T_BFS))):
        FNR_mu = []
        FNR_95 = []
        FNR_05 = []
        for p in p_uniq:
            FNR_mu.append(np.median(FPR_BFS[(T_BFS == t) & (p_arr_BFS == p)]))
            FNR_95.append(np.percentile(FPR_BFS[(T_BFS == t) & (p_arr_BFS == p)], 95))
            FNR_05.append(np.percentile(FPR_BFS[(T_BFS == t) & (p_arr_BFS == p)], 5))
        FNR_mu = np.array(FNR_mu)
        FNR_95 = np.array(FNR_95)
        FNR_05 = np.array(FNR_05)
        ax.plot(p_uniq, FNR_mu, label=f"Duration of Pandemic = {t}, Full Search")
        ax.fill_between(p_uniq, FNR_05, FNR_95, alpha=0.1)
    #ax.set_ylabel("Pandemic Transmission Time")
    if log_scale:
        ax.set_yscale("log")
    #ax.set_title(f"False Positive Rates (BFS) = FalsePos / {denominator}")
    for t in sorted(list(set(T_DFS))):
        FPR_mu = []
        FPR_95 = []
        FPR_05 = []
        for p in p_uniq:
            FPR_mu.append(np.median(FPR_DFS[(T_DFS == t) & (p_arr_DFS == p)]))
            FPR_95.append(np.percentile(FPR_DFS[(T_DFS == t) & (p_arr_DFS == p)], 95))
            FPR_05.append(np.percentile(FPR_DFS[(T_DFS == t) & (p_arr_DFS == p)], 5))
        FPR_mu = np.array(FPR_mu)
        FPR_95 = np.array(FPR_95)
        FPR_05 = np.array(FPR_05)
        ax.plot(p_uniq, FPR_mu, label=f"Duration of Pandemic = {t}, Geometric Search")
        ax.fill_between(p_uniq, FPR_05, FPR_95, alpha=0.1)
    ax.set_xlabel("Infection Probability")
    #ax.set_ylabel("Pandemic Transmission Time")
    ax.set_ylabel("Misclassified as Boundaries")
    if log_scale:
        ax.set_yscale("log")
    #ax.set_title(f"False Positive Rates (DFS) = FalsePos / {denominator}Num")
    ax.legend()
    plt.savefig(f"../Plots/Deliverable/pop={pop}_non-coop-prob={non_coop_prob}_radius={radius}_bar={bar}_log={log_scale}_type={type}_FalsePos.tiff", dpi=300)
    plt.clf()
    plt.close()

    fig, ax = plt.subplots()
    for t in sorted(list(set(T_BFS))):
        Test_mu = []
        Test_95 = []
        Test_05 = []
        for p in p_uniq:
            Test_mu.append(np.median(TestRatio_BFS[(T_BFS == t) & (p_arr_BFS == p)]))
            Test_95.append(np.percentile(TestRatio_BFS[(T_BFS == t) & (p_arr_BFS == p)], 95))
            Test_05.append(np.percentile(TestRatio_BFS[(T_BFS == t) & (p_arr_BFS == p)], 5))
        Test_mu = np.array(Test_mu)
        Test_95 = np.array(Test_95)
        Test_05 = np.array(Test_05)
        ax.plot(p_uniq, Test_mu, label=f"Duration of Pandemic = {t}, Full Search")
        ax.fill_between(p_uniq, Test_05, Test_95, alpha=0.1)
    #ax.set_ylabel("Pandemic Transmission Time")
    if log_scale:
        ax.set_yscale("log")
    #ax.set_title(f"# Test Conducted (BFS) / {denominator}Num")
    
    #ax.plot_trisurf(p, T, TestRatio)
    for t in sorted(list(set(T_DFS))):
        Test_mu = []
        Test_95 = []
        Test_05 = []
        for p in p_uniq:
            Test_mu.append(np.median(TestRatio_DFS[(T_DFS == t) & (p_arr_DFS == p)]))
            Test_95.append(np.percentile(TestRatio_DFS[(T_DFS == t) & (p_arr_DFS == p)], 95))
            Test_05.append(np.percentile(TestRatio_DFS[(T_DFS == t) & (p_arr_DFS == p)], 5))
        Test_mu = np.array(Test_mu)
        Test_95 = np.array(Test_95)
        Test_05 = np.array(Test_05)
        ax.plot(p_uniq, Test_mu, label=f"Duration of Pandemic = {t}, Geometric Search")
        ax.fill_between(p_uniq, Test_05, Test_95, alpha=0.1)
    ax.set_xlabel("Infection Probability")
    #ax.set_ylabel("Pandemic Transmission Time")
    ax.set_ylabel(f"# Test Conducted / N")
    if log_scale:
        ax.set_yscale("log")
    #ax.set_title(f"# Test Conducted (DFS) / {denominator}Num")
    ax.legend()
    
    plt.savefig(f"../Plots/Deliverable/pop={pop}_non-coop-prob={non_coop_prob}_radius={radius}_bar={bar}_log={log_scale}_type={type}_TestNum.tiff", dpi=300)
    plt.clf()
    plt.close()

#    if non_coop_prob == 0:
#        plt.suptitle("Population = " + str(pop) + " NonCooperativeRate = " + str(non_coop_prob) + "\nType = " + str(type))
#    else:
#        plt.suptitle("Population = " + str(pop) + " NonCooperativeRate = " + str(non_coop_prob) + "\nRadius = " + str(radius) + " Bar = " + str(bar) + "\nType = " + str(type))

non_coop_prob_lst = list(set(dt["NonCooperativeProb"]))
radius_lst = [x for x in list(set(dt["Radius"])) if x > 0]
pop_lst = list(set(dt["Population"]))

#for pop in pop_lst:
#    for non_coop_prob in non_coop_prob_lst:
#        for type in ["Cell", "People"]:
#            for log in [True, False]:
#                non_coop_prob = round(non_coop_prob, 3)
#                if non_coop_prob == 0:
#                    plot_2D(0, non_coop_prob, 0, pop, type, log)
#                else:
#                    for radius in radius_lst:
#                        for bar in [0, 0.5]:
#                            plot_2D(bar, non_coop_prob, radius, pop, type, log)
#for pop in pop_lst:
#    for non_coop_prob in [0, 0.333]:
#        non_coop_prob = round(non_coop_prob, 3)
#        if non_coop_prob == 0:
#            plot_2D(0, non_coop_prob, 0, pop, "People", False)
#        else:
#            plot_2D(0.5, non_coop_prob, 1, pop, "People", False)

plot_2D_V2(0, 0, 0, 250000, "People", False)
plot_2D_V2(0, 0, 0, 1000000, "People", False)
plot_2D_V2(0.5, 0.333, 2, 250000, "People", False)
plot_2D_V2(0.5, 0.333, 2, 1000000, "People", False)
#plot_2D_V2(0, 0, 0, 250000, "People", True)
#plot_2D_V2(0, 0, 0, 1000000, "People", True)
#plot_2D_V2(0.5, 0.333, 2, 250000, "People", True)
#plot_2D_V2(0.5, 0.333, 2, 1000000, "People", True)
