from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir

def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inx, data_set, labels, k):
    data_set_size = data_set.shape[0]
    #print("data_set_size {}".format(data_set_size))

    diff_mat = tile(inx, (data_set_size, 1)) - data_set
    #print("diff_mat {}".format(diff_mat))

    square_diff_mat = diff_mat ** 2
    #print("square_diff_mat {}".format(square_diff_mat))

    square_distance = square_diff_mat.sum(axis=1)
    #print("square_distance {}".format(square_distance))

    distance = square_distance ** 0.5
    #print("distance {}".format(distance))

    sorted_distance_indicies = distance.argsort()
    #print("sorted_distance_indicies {}".format(sorted_distance_indicies))

    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_distance_indicies[i]]
        if vote_label in class_count:
            class_count[vote_label] = class_count[vote_label] + 1
        else:
            class_count[vote_label] = 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    #print("sorted_class_count {}".format(sorted_class_count))
    return sorted_class_count[0][0]


def file_2_matrix(file_name):
    file_reader = open(file_name)
    file_lines = file_reader.readlines()
    no_of_lines = len(file_lines)

    return_matrix = zeros((no_of_lines, 3))
    class_labels_vector = []
    index = 0
    for line in file_lines:
        line = line.strip()
        list_from_line = line.split('\t')
        return_matrix[index, :] = list_from_line[0:3]
        class_labels_vector.append(int(list_from_line[-1]))
        index += 1
    return return_matrix, class_labels_vector


def auto_norm(data_set):
    min_value = data_set.min(0)
    max_value = data_set.max(0)
    #print("min {}, max {}".format(min_value, max_value))

    ranges = max_value - min_value
    norm_data_set = zeros(shape(data_set))
    #print(norm_data_set)

    m = data_set.shape[0]
    norm_data_set = data_set - tile(min_value, (m, 1))
    norm_data_set = norm_data_set/tile(ranges, (m, 1))
    return norm_data_set, ranges, min_value


def dating_class_test():
    horatio = 0.10
    dating_matrix, dating_class_labels = file_2_matrix("datingTestSet2.txt")
    normal_dating_matrix, ranges, min_value = auto_norm(dating_matrix)
    m = normal_dating_matrix.shape[0]
    number_test = int(m * horatio)
    error_count = 0.0
    for i in range(number_test):
        classify_result = classify0(normal_dating_matrix[i, :], normal_dating_matrix[number_test:m, :], dating_class_labels[number_test:m], 3)
        print("the classify result is {}, the real is {}".format(classify_result, dating_class_labels[i]))
        if classify_result != dating_class_labels[i]:
            error_count += 1.0
    print("error ratio is {}".format(error_count/float(number_test)))


def image_2_vector(file_name):
    return_vector = zeros((1, 1024))
    fr = open(file_name)
    for i in range(32):
        line_string = fr.readline()
        for j in range(32):
            return_vector[0, 32*i+j] = int(line_string[j])
    return return_vector

def hand_writing_class_test():
    hw_labels = []
    training_files_list = listdir("trainingDigits")
    m = len(training_files_list)
    training_matrix = zeros((m, 1024))
    for i in range(m):
        file_name = training_files_list[i]
        file_str = file_name.split('.')[0]
        class_number = int(file_str.split('_')[0])
        hw_labels.append(class_number)
        training_matrix[i, :] = image_2_vector("trainingDigits/{}".format(file_name))

    test_file_list = listdir("testDigits")
    error_count = 0.0
    mtest = len(test_file_list)
    for i in range(mtest):
        file_name = test_file_list[i]
        file_str = file_name.split('.')[0]
        class_number = int(file_str.split('_')[0])
        test_matrix = image_2_vector("testDigits/{}".format(file_name))
        classify_result = classify0(test_matrix, training_matrix, hw_labels, 1)
        print("classify result is {}, real is {}".format(classify_result, class_number))
        if classify_result != class_number:
            error_count += 1.0

    print("error ratio is {}".format(error_count/float(mtest)))


if __name__ == "__main__":
    #dating_matrix, dating_class = file_2_matrix("datingTestSet2.txt")
    #print(dating_matrix)
    #print(dating_class)

    #norm_data_set, ranges, min_value = auto_norm(dating_matrix)
    #print(norm_data_set)
    #dating_class_test()
    hand_writing_class_test()

    #draw
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #print(tile(array([15.0]),  len(dating_class)))
    #print(array(dating_class))
    #point_color = multiply(array(dating_class), tile(array([15]),  len(dating_class)))
    #print(point_color)
    #ax.scatter(dating_matrix[:, 1], dating_matrix[:, 2], 15.0*array(dating_class), 15.0*array(dating_class))
    #plt.show()
#    group, labels = create_data_set()
#    classify0([0, 0], group, labels, 3)