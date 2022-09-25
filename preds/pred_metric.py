fred = open("testMydata.txt", "r")
correct_nolack = 0
nolack_num = 0
correct_stack = 0  #真正的 stack的数目
stack_num = 0 #预测出的 stack的数目
correct_newbranch = 0
newbranch_num = 0
correct_post = 0
post_num = 0
correct_huiliu = 0
huiliu_num = 0
correct_type = 0 #正确的类型数
correct_num = 0 #正确的个数
lack_correct = 0 #缺失地方正确的个数
slack_num = 0 #真正缺失的个数
plack_num = 0  #预测出缺失的个数
total_num = 0  #所有的个数
em = 0
em_num = 0
for line in fred.readlines():
    if line == '\n':
        total_num += 1
        cur_pred_type = None
        cur_tar_type = None
        continue
    else:
        content = line.strip().split(" ")
        if int(content[0]) == 0:
            cur_tar_type = content[1]
            cur_pred_type = content[2]
            #预测出的缺的stack的数目
            if content[1] == 'stack':
                stack_num += 1
            elif content[1] == 'postposition':
                post_num += 1
            elif content[1] == 'influx_chunks':
                huiliu_num += 1
            elif content[1] == 'new_branch':
                newbranch_num += 1
            else:
                nolack_num += 1

            if cur_tar_type == cur_pred_type:
                correct_type += 1
                if cur_tar_type == 'nolack':
                    correct_num += 1

        else:
            if content[1] != 'None':
                slack_num += 1
                em_num += 1
            if content[2] != 'None':
                plack_num += 1
            if content[1] == content[2]:
                correct_num += 1
                lack_correct += 1

                if cur_tar_type == 'stack':
                    correct_stack += 1
                elif cur_tar_type == 'postposition':
                    correct_post += 1
                elif cur_tar_type == 'influx_chunks':
                    correct_huiliu += 1
                elif cur_tar_type == 'new_branch':
                    correct_newbranch += 1
                else:
                    correct_nolack += 1

            if content[1] != 'None' and content[2] != 'None' :
                flag = 1
                for i in content[1]:
                    if i not in content[2]:
                        flag = 0
                if flag:
                    em += 1

emzhi = em / em_num

stack_acc = correct_stack / (stack_num+ 1e-8)
newbranch_acc = correct_newbranch / (newbranch_num + 1e-8)
post_acc = correct_post / (post_num + 1e-8)
huiliu_acc = correct_huiliu / (huiliu_num + 1e-8)

acc_type = correct_type / total_num
lack_p = lack_correct / plack_num
lack_r = lack_correct / slack_num
lack_f1 = 2*lack_p*lack_r/(lack_p + lack_r)

acc_all_content = correct_num / total_num


print("em值\t\t\t{}".format(emzhi))
print("stack_acc\t\t{}".format(stack_acc))
print("newbranch_acc\t{}".format(newbranch_acc))
print("post_acc\t\t{}".format(post_acc))
print("huiliu_acc\t\t{}".format(huiliu_acc))
print("acc_type\t\t{}".format(acc_type))
print("lack_p\t\t{}".format(lack_p))
print("lack_r\t\t{}".format(lack_r))
print("lack_f1\t\t{}".format(lack_f1))
print("acc_all_content\t{}".format(acc_all_content))



'''
my:
em值			0.7322456813819578
stack_acc		0.7228260869565217
newbranch_acc	0.3577235772212307
post_acc		0.617647058732699
huiliu_acc		0.4615384614497041
acc_type		0.9270223997717221
lack_p		0.6056933483652762
lack_r		0.6874600127959053
lack_f1		0.6439916092298471
acc_all_content	0.8872877728634613
'''


'''
liu
em值			0.6774337866857552
stack_acc		0.657296179598267
newbranch_acc	0.8728956228882754
post_acc		0.8637274548925105
huiliu_acc		0.9150141642800279
acc_type		0.8848715396283656
lack_p		0.6629016723579372
lack_r		0.6774337866857552
lack_f1		0.6700889498606009
acc_all_content	0.8557427948426242
'''
