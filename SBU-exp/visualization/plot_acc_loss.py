import re
import matplotlib

#matplotlib.use('Agg')
import matplotlib.pyplot as plt


def loss_graph(file):
    regex_total_loss = re.compile(r"(filter\sloss:\s)(-?\d+\.\d+)")
    regex_utility_loss = re.compile(r"(utility\sloss:\s)(-?\d+\.\d+)")
    regex_budget_loss = re.compile(r"(budget\sloss:\s)(-?\d+\.\d+)")
    step_lst = []
    step = 0
    total_loss_lst = []
    utility_loss_lst = []
    budget_loss_lst = []
    with open(file) as f:
        for line in f:
            line = line.rstrip('\n')
            r_total = re.search(regex_total_loss, line)
            r_utility = re.search(regex_utility_loss, line)
            r_budget = re.search(regex_budget_loss, line)
            if r_total is not None and r_utility is not None and r_budget is not None:
                total_loss_lst.append(r_total.group(2))
                utility_loss_lst.append(r_utility.group(2))
                budget_loss_lst.append(r_budget.group(2))
                step_lst.append(step)
                step += 1
            else:
                continue

    print(step_lst)
    plt.plot(step_lst, total_loss_lst, 'go-', label='Filter Loss')
    plt.plot(step_lst, utility_loss_lst, 'rx-', label='Utility Loss')
    plt.plot(step_lst, budget_loss_lst, 'b^-', label='Budget Loss')

    plt.legend(loc='upper right', shadow=True)
    plt.xlabel('Step#')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
    #plt.savefig(fname='loss.png')

def acc_gap_graph(file0, file1):
    regex_utility = re.compile(r"(utility accuracy:\s)(\d+\.\d+)")
    regex_budget = re.compile(r"(budget accuracy:\s)(\d+\.\d+)")
    tr_step_lst = []
    tr_utility_acc_lst = []
    tr_budget_acc_lst = []

    step = 0
    with open(file0) as f:
        for line in f:
            line = line.rstrip('\n')
            r_utility = re.search(regex_utility, line)
            r_budget = re.search(regex_budget, line)
            if r_utility is not None and r_budget is not None:
                tr_utility_acc_lst.append(float(r_utility.group(2)))
                tr_budget_acc_lst.append(float(r_budget.group(2)))
                tr_step_lst.append(step)
                step += 1
    #print(step_lst)
    val_step_lst = []
    val_utility_acc_lst = []
    val_budget_acc_lst = []

    step = 0
    with open(file1) as f:
        for line in f:
            line = line.rstrip('\n')
            r_utility = re.search(regex_utility, line)
            r_budget = re.search(regex_budget, line)
            if r_utility is not None and r_budget is not None:
                val_utility_acc_lst.append(float(r_utility.group(2)))
                val_budget_acc_lst.append(float(r_budget.group(2)))
                val_step_lst.append(step)
                step += 1
    step_lst = val_step_lst
    utility_acc_lst = [a_i - b_i for a_i, b_i in zip (tr_utility_acc_lst, val_utility_acc_lst)]
    budget_acc_lst = [a_i - b_i for a_i, b_i in zip(tr_budget_acc_lst, val_budget_acc_lst)]
    plt.plot(step_lst, utility_acc_lst, 'rx-', label='Utility Task Gap')
    plt.plot(step_lst, budget_acc_lst, 'b^-', label='Budget Task Gap')
    plt.legend(loc='center right', shadow=True)
    plt.xlabel('Step#')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Gap Between Training and Validation')
    plt.show()

def acc_graph(file):
    regex_utility = re.compile(r"(utility accuracy:\s)(\d+\.\d+)")
    regex_budget = re.compile(r"(budget accuracy:\s)(\d+\.\d+)")
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
                budget_acc_lst.append(float(r_budget.group(2)))
                step_lst.append(step*5)
                step += 1

    plt.plot(step_lst, utility_acc_lst, 'rx-', label='Utility Task Acc')
    plt.plot(step_lst, budget_acc_lst, 'b^-', label='Budget Task Acc')

    plt.legend(loc='center right', shadow=True)
    plt.xlabel('Step#')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.show()

if __name__ == '__main__':
    acc_graph('../summaries/L1LossNoLambdaDecayAvgReplicateMonitorBudgetMonitorUtilityResample8_2.0_0.5_SuppressingMostConfident_20180429-112811/test_summary.txt')