def view_bar(step,total_nums, epoch,epoch_num):
    rate = step/total_nums
    rate_num = int(rate*40)
    r = '\r[%s%s]%d%%\t step-%d/%d epoch-%d/%d'%('>'*rate_num,'-'*(40-rate_num),rate*100,step,total_nums,epoch,epoch_num)
    print(r,end='',flush=True)