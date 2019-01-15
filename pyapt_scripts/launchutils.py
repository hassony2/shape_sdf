def clean_args(dic):
    final_dic = {}
    for key, val in dic.items():
        if not (val is False or val is None):
            final_dic[key] = val
    return final_dic
