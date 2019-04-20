import glob
import json

# modify the root to ModelNet40
root = "./ModelNet40"

clses = []


def mk_model_net40():
    l = glob.glob(root+'/*')

    # save classes
    for x in l:
        clses.append(x.split('/')[-1])
    with open("./classes.txt", 'w') as f:
        for i, x in enumerate(clses):
            f.write(str(i)+":\t"+x+"\n")

    # generate train single
    train_single_3d = []
    for i, path in enumerate(l):
        single_list = glob.glob(path+'/train/*')
        for single_path in single_list:
            train_single_3d.append([single_path, i])
    with open("train_single_3d.json", 'w') as f:
        json.dump(train_single_3d, f)

    # generate test single
    test_single_3d = []
    for i, path in enumerate(l):
        single_list = glob.glob(path + '/test/*')
        for single_path in single_list:
            test_single_3d.append([single_path, i])
    with open("test_single_3d.json", 'w') as f:
        json.dump(test_single_3d, f)

    # generate train multi
    # ./ModelNet40/xbox/train/xbox_0027.obj_whiteshaded_v0.png
    train_3d = []
    for i, path in enumerate(l):
        single_list = glob.glob(path + '/train/*')
        tmpl = []
        for single_path in single_list:
            tmpl.append(single_path.split("obj_whiteshaded")[-2])
        tmpl = list(set(tmpl))
        tmpl = [[x, i] for x in tmpl]
        train_3d.extend(tmpl)
    with open("train_3d.json", 'w') as f:
        json.dump(train_3d, f)

    # generate test multi
    # ./ModelNet40/xbox/train/xbox_0027.obj_whiteshaded_v0.png
    test_3d = []
    for i, path in enumerate(l):
        single_list = glob.glob(path + '/test/*')
        tmpl = []
        for single_path in single_list:
            tmpl.append(single_path.split("obj_whiteshaded")[-2])
        tmpl = list(set(tmpl))
        tmpl = [[x, i] for x in tmpl]
        test_3d.extend(tmpl)
    with open("test_3d.json", 'w') as f:
        json.dump(test_3d, f)


if __name__ == '__main__':
    mk_model_net40()
