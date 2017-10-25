import config

def get_class_to_label():
    class_to_label = {}
    for i in range(len(config.classnames[0])):
        class_to_label[config.classnames[0][i][0][3:]] = i

    return class_to_label

def get_label_to_class():
    return {val: key for key, val in get_class_to_label().items()}
