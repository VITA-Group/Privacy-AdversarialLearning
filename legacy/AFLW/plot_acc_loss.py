import re
import matplotlib

#matplotlib.use('Agg')
import matplotlib.pyplot as plt



def acc_loss_graph(file):
    regex_utility = re.compile(r"(validation utility loss:\s)(\d+\.\d+)")
    regex_budget = re.compile(r"(validation budget accuracy:\s)(\d+\.\d+)")
    step_lst = []
    utility_acc_lst = []
    budget_acc_lst = []
    step = 0
    with open(file) as f:
        for line in f:
            line = line.rstrip('\n')
            r_utility = re.search(regex_utility, line)
            r_budget = re.search(regex_budget, line)
            if r_utility is not None and r_budget is not None:
                utility_acc_lst.append(float(r_utility.group(2)))
                #utility_acc_lst.append(float(r_utility.group(2)) / 1000)
                budget_acc_lst.append(float(r_budget.group(2)))
                step_lst.append(step)
                step += 25

    plt.plot(step_lst, utility_acc_lst, 'rx-', label='Utility Task Loss')
    #plt.plot(step_lst, budget_acc_lst, 'b^-', label='Budget Task Acc')
    plt.legend(loc='center right', shadow=True)
    plt.xlabel('Step#')
    plt.ylabel('Loss')
    plt.title('Validation Loss of Head Pose Estimation')
    plt.show()

if __name__ == '__main__':
    acc_loss_graph('/home/wuzhenyu/Desktop/summaries/NoL1LossLambdaDecayAvgReplicateMonitorBudgetNoMonitorUtilityRestart4_100.0_0.5_SuppressingMostConfident_20180629-184804/val_summary.txt')