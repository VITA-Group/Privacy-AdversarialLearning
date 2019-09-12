
# coding: utf-8

# In[32]:


import os

filenames = []
class_dict = {}
video_dir = 'hmdb51_org'
i = 0
for path, subdirs, files in os.walk(video_dir):
    if files:
        class_dict[path.split('/')[-1]] = i
        i += 1
    for name in files:
        filenames.append(os.path.join(path,name))

video_class_dict = {}
for path, subdirs, files in os.walk(video_dir):
    for name in files:
        video_class_dict[name] = path.split('/')[-1]


# In[22]:


train_lst = []
test_lst = []
nouse_lst = []
for file in os.listdir('split1'):
    with open('split1/'+file, "r") as f:
        for line in f:
            name, id = line.split()
            if id == '1':
                train_lst.append(name)
            elif id == '2':
                test_lst.append(name)
            elif id == '0':
                nouse_lst.append(name)
            else:
                raise ValueError('Error!')


# In[35]:


with open('classInd.txt', 'w') as f:
    for k,v in class_dict.items():
        f.write(str(v+1)+'\t'+k+'\n')


# In[33]:


with open('trainlist01.txt','w') as f:
    for name in train_lst:
        if name not in video_class_dict:
            print(name + " not in")
        else:
            f.write(video_class_dict[name] + '/' + name + '\n')


# In[34]:


with open('testlist01.txt', 'w') as f:
    for name in test_lst:
        if name not in video_class_dict:
            print(name + " not in")
        else:
            f.write(video_class_dict[name] + '/' + name + '\n')

