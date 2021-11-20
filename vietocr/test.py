# # import torch

# # # original saved file with DataParallel
# # state_dict = torch.load('/home/ancv/Work/transformerocr.pth', map_location="cuda:0")
# # # create new OrderedDict that does not contain `module.`
# # from collections import OrderedDict
# # count = 0
# # new_state_dict = OrderedDict()
# # for k, v in state_dict.items():
# #     name = k[7:] # remove `module.`
# #     print(k)
# #     new_state_dict[name] = v
# #     count += 1
# # print(count)


# # load params
# # model.load_state_dict(new_state_dict)
# from khmernltk import word_tokenize
# import re


# a = "១៩៥"


# tokens = a.split(" ")
# result = []
# for d, token in enumerate(tokens):
#     if re.search("[a-zA-Z0-9<.]", token) is None:
#         temp = word_tokenize(token, return_tokens=True)
#         print(token, temp)
#         temp = [t for t in temp if t not in ["", " "]]
        
#         # split number in each token
#         for i in range(len(temp)):
#             res = []
#             txt = temp[i]
#             print(txt)
#             for num in ["០", "១", "២", "៣", "៤", "៥", "៦", "៧", "៨", "៩"]:
#                 indices = [i for i, x in enumerate(txt) if x == num]
#                 res += indices
#             if len(res) > 0:
#                 temp[i] = []
#                 res.sort()
#                 print(res)
#                 if res[0] > 0: temp[i].append(txt[:res[0]])
#                 for j in range(len(res)):
#                     temp[i].append(txt[res[j]])
#                     if res[j] + 1 < len(res) and res[j+1] > (res[j] + 1): temp[j].append(txt[res[j]+1:res[j+1]])
#                 if res[-1] + 1 < len(txt): result.append(txt[res[-1] + 1:])
#             else:
#                 temp[i] = txt
        
#         for t in temp:
#             if isinstance(t, list):
#                 for j in t: result.append(j)
#             else: result.append(t)
        
#         if d < len(tokens) - 1:
#             result.append(" ")
#     else:
#         for c in token: result.append(c)

    

# print(result)

import os, shutil

path = "/home/ancv/Work/KhmOcr/real_data_test/crop_box/"
out = "/home/ancv/Work/KhmOcr/real_data_test/mrz_box_test/"
with open("predict_latin_transfomer.log", "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split(" || ")
        file = line[0].strip()
        shutil.copy(path + file, out)