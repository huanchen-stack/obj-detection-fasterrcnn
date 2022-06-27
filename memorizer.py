from sigfig import round
class MemRec(object):

    def __init__(self):
        super().__init__()
        self.cpu = None
        self.cuda = None

    def get_mem_helper(self, ret_list, col_name):
        if col_name not in ret_list[0]:
            return 0.0
        index = ret_list[0].index(col_name)
        mem_all_layers_list = [lis[index] for lis in ret_list]
        mem_all_layers_list.pop(0)
        mem_all_layers = 0
        for item in mem_all_layers_list:
            mem, unit = item.rstrip().split()
            # target unit is Kb
            mem = float(mem)
            if unit == "b":
                mem *= 0.000001
            elif unit == "Kb":
                mem *= 0.001
            elif unit == "Mb":
                mem *= 1
            elif unit == "Gb":
                mem *= 1000
            else:
                print(unit)
                exit(1)
            mem_all_layers += mem
        return round(mem_all_layers, sigfigs=4)

    def get_mem(self, prof_report, usingcuda=True):

        ret_list = parse_prof_table(prof_report)

        if usingcuda:
            return self.get_mem_helper(ret_list, "CPU Mem"), self.get_mem_helper(ret_list, "CUDA Mem")
        else:
            return self.get_mem_helper(ret_list, "CPU Mem")



def parse_prof_table(prof_report):

    ret_list = []

    flip = False
    parsing_str = prof_report[0]
    parsing_idx = []
    for i in range(len(parsing_str)):
        if parsing_str[i] == '-':
            flip = True
        if flip and parsing_str[i] == ' ':
            parsing_idx.append(i)
            flip = False

    head_str_list = []
    parsing_str = prof_report[1]
    head_str = ""
    for i in range(len(parsing_str)):
        if i-1 in parsing_idx:
            head_str_list.append(head_str.lstrip().rstrip())
            head_str = ""
        else:
            head_str += parsing_str[i:i+1]
    
    ret_list.append(head_str_list)

    parsing_str_list = prof_report[3:-4]
    for parsing_str in parsing_str_list:
        head_str_list = []
        head_str = ""
        for i in range(len(parsing_str)):
            if i-1 in parsing_idx:
                head_str_list.append(head_str.lstrip().rstrip())
                head_str = ""
            else:
                head_str += parsing_str[i:i+1]
        ret_list.append(head_str_list)

    return ret_list
