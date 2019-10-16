import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt

# --------------------------------------------CONSTANTS--------------------------------------------
# FILE_TYPES
FILE_TYPE_CSV = 'csv'
FILE_TYPE_PKL = 'pkl'

# CRITERIA_TYPES
CRITERIA_TYPE_CENTER_COORDINATES = 'center'
CRITERIA_TYPE_AREA = 'area'

# COMPARISON_PARAMETERS
ITERATIONS_NUMBER = 500
MAX_CENTER_DISPLACEMENT = 0.5
MIN_CENTER_DISPLACEMENT = 0.0
MAX_AREA_DIFFERENCE = 1.0
MIN_AREA_DIFFERENCE = 0.0

# --------------------------------------------INPUT--------------------------------------------

# INPUT_DATA
image_number = 2
file_type = FILE_TYPE_CSV

# COMARISON_CRITERIA
cnts_center_displacement = 0.3
cnts_area_difference_factor = 0.3
#criteria = CRITERIA_TYPE_CENTER_COORDINATES
criteria = CRITERIA_TYPE_AREA

if image_number == 1:
    FILE_NAME_IMAGE_INITIAL = 'img1.jpg'
    FILE_NAME_IMAGE_GROUND_TRUTH = 'cells1.jpg'
    FILE_NAME_FILE_CSV = 'csv1.txt'
    FILE_NAME_FILE_PKL = ''
elif image_number == 2:
    FILE_NAME_IMAGE_INITIAL = 'img2_2016-03-01 21.42.11.jpg'
    FILE_NAME_IMAGE_GROUND_TRUTH = 'cells2_2016-03-01 21.42.11.jpg'
    FILE_NAME_FILE_CSV = 'csv2.txt'
    FILE_NAME_FILE_PKL = 'cells1.pkl'
elif image_number == 3:
    FILE_NAME_IMAGE_INITIAL = 'img3_20160630_160547.jpg'
    FILE_NAME_IMAGE_GROUND_TRUTH = 'cells3_20160630_160547.jpg'
    FILE_NAME_FILE_CSV = 'csv3.txt'
    FILE_NAME_FILE_PKL = 'cells2_2016-03-01_21.42.11.pkl'
elif image_number == 4:
    FILE_NAME_IMAGE_INITIAL = 'img4_20160630_160548.jpg'
    FILE_NAME_IMAGE_GROUND_TRUTH = 'cells4_20160630_160548.jpg'
    FILE_NAME_FILE_CSV = 'csv4.txt'
    FILE_NAME_FILE_PKL = 'cells4_20160630_160548.pkl'
else:
    print('INCORRECT NUMBER!')

PATH_TO_FOLDER_IMAGE_INITIAL = '../resources/images_initial/'
PATH_TO_FOLDER_IMAGE_GROUND_TRUTH = '../resources/images_photoshop/'
PATH_TO_FOLDER_FILE_CSV = '../resources/csv/'
PATH_TO_FOLDER_FILE_PKL = 'r../resources/pkl/'

PATH_TO_IMAGE_INITIAL = PATH_TO_FOLDER_IMAGE_INITIAL + FILE_NAME_IMAGE_INITIAL
PATH_TO_IMAGE_GROUND_TRUTH = PATH_TO_FOLDER_IMAGE_GROUND_TRUTH + FILE_NAME_IMAGE_GROUND_TRUTH
PATH_TO_FILE_CSV = PATH_TO_FOLDER_FILE_CSV + FILE_NAME_FILE_CSV
PATH_TO_FILE_PKL = PATH_TO_FOLDER_FILE_PKL + FILE_NAME_FILE_PKL

# --------------------------------------------OUTPUT--------------------------------------------
PATH_TO_FOLDER_OUTPUT = '../output/'
PATH_TO_FOLDER_OUTPUT_MATPLOTLIB_PLOTS = '../output/matplotlib_plots/'

PATH_TO_FOLDER_OUTPUT_GT_CONTOURS = '../output/GT_contours/'
PATH_TO_FOLDER_OUTPUT_GT_ELLIPSES = '../output/GT_ellipses/'
PATH_TO_FOLDER_OUTPUT_METHOD_CONTOURS = '../output/method_cnts/'
PATH_TO_FOLDER_OUTPUT_VALIDATION = '../output/validation/'

PATH_TO_FILE_OUTPUT_GT_CONTOURS = PATH_TO_FOLDER_OUTPUT_GT_CONTOURS + 'GT_contours_' + FILE_NAME_IMAGE_INITIAL
PATH_TO_FILE_OUTPUT_GT_ELLIPSES = PATH_TO_FOLDER_OUTPUT_GT_ELLIPSES + 'GT_ellipses_' + FILE_NAME_IMAGE_INITIAL
PATH_TO_FILE_OUTPUT_METHOD_CONTOURS = PATH_TO_FOLDER_OUTPUT_METHOD_CONTOURS + 'method_cnts_' + FILE_NAME_IMAGE_INITIAL
PATH_TO_FILE_OUTPUT_VALIDATION = PATH_TO_FOLDER_OUTPUT_VALIDATION + 'validation_' + FILE_NAME_IMAGE_INITIAL


def generate_contours_list_from_file(path_to_file, file_type=FILE_TYPE_CSV):
    contours_from_file = []

    if file_type == FILE_TYPE_CSV:
        lines = open(path_to_file).read().split('\n')
        for line in lines:
            firstPointFlag = True
            contour_points = np.empty((1, 2), dtype=np.int32)
            numbers = line.split(' ')
            i = 0
            while i < len(numbers) - 4:
                if numbers[i] != '':
                    x = int(numbers[i].strip())
                    x //= 2
                    y = int(numbers[i + 1].strip())
                    y //= 2
                    if firstPointFlag == True:
                        contour_points[0] = [x, y]
                        firstPointFlag = False
                    else:
                        contour_points = np.append(contour_points, [[x, y]], axis=0)
                i += 2
            contours_from_file.append(contour_points)
    elif file_type == FILE_TYPE_PKL:
        # with open(r'./resources/pkl/20160630_160547.pkl', 'rb') as file:
        with open(path_to_file, 'rb') as file:
            contours = pickle.load(file)

        # print(contours)
        contours_from_file = contours

    return contours_from_file


def generate_contours_list_from_GROUND_TRUTH_file(image, title):
    image_copy = image.copy()

    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(img_gray, 20, 255, 0)

    kernel = np.ones((2, 2), np.uint8)

    dilation = cv2.dilate(thresh, kernel, iterations=1)
    dilation = cv2.dilate(dilation, kernel, iterations=1)
    dilation = cv2.dilate(dilation, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    new_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50000:
            new_contours.append(cnt)
            if len(cnt) > 4:
                ellipse = cv2.fitEllipse(cnt)
                image_copy = cv2.ellipse(image_copy, ellipse, (255, 0, 0), 2)

    # cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    # cv2.imshow(title, image_copy)
    #
    # PATH_TO_SAVE_FILE = PATH_TO_FOLDER_OUTPUT + str(title) + '_' + file_type + '.png'
    # cv2.imwrite(PATH_TO_SAVE_FILE, image_copy)

    return new_contours


def convert_color(color):
    if color == 'B':
        cnt_color = (255, 0, 0)
    elif color == 'G':
        cnt_color = (0, 255, 0)
    elif color == 'R':
        cnt_color = (0, 0, 255)
    elif color == 'Y':
        cnt_color = (0, 255, 255)

    return cnt_color


def draw_contours_on_image(contours, color, image, title, isFilled=False):
    image_copy = image.copy()

    cnt_color = convert_color(color)

    for cnt in contours:
        if isFilled == False:
            cv2.drawContours(image_copy, [cnt], 0, cnt_color, 2)
        if isFilled == True:
            cv2.drawContours(image_copy, [cnt], 0, cnt_color, -1)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, image_copy)

    PATH_TO_SAVE_FILE = PATH_TO_FOLDER_OUTPUT + str(title) + '_' + file_type + '.png'
    cv2.imwrite(PATH_TO_SAVE_FILE, image_copy)


def draw_contours_on_image_from_list_of_dict(list_of_contours_dict, color, image, title):
    image_copy = image.copy()

    cnt_color = convert_color(color)

    for cnt_dict in list_of_contours_dict:
        if cnt_dict['is_found'] == True:
            cnt = cnt_dict['cnt']
            cv2.drawContours(image_copy, [cnt], 0, cnt_color, -1)
        else:
            cnt = cnt_dict['cnt']
            cv2.drawContours(image_copy, [cnt], 0, cnt_color, 2)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, image_copy)

    PATH_TO_SAVE_FILE = PATH_TO_FOLDER_OUTPUT + str(title) + '_' + file_type + '.png'
    cv2.imwrite(PATH_TO_SAVE_FILE, image_copy)


def find_cnt_center(contour):
    M = cv2.moments(contour)
    if M['m00'] == 0:
        cx = int(M['m10'] / (M['m00'] + 0.000001))
        cy = int(M['m01'] / (M['m00'] + 0.000001))
    else:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    return cx, cy


def get_list_of_cnt_data_dict(contours, color):
    list = []
    id = 0
    for cnt in contours:
        dict = {}

        dict['cnt_id'] = id
        dict['cnt_color'] = convert_color(color)
        dict['cnt_center'] = find_cnt_center(cnt)
        dict['cnt_area'] = cv2.contourArea(cnt)
        dict['is_found'] = False
        dict['cnt'] = cnt

        list.append(dict)
        id += 1

    return list


def get_resized_image_for_validation(path_to_file):
    image_init_1 = cv2.imread(PATH_TO_IMAGE_INITIAL)
    image_init_1 = cv2.resize(image_init_1, None, fx=0.5, fy=0.5)
    height1, width1, _ = image_init_1.shape

    image_cells = cv2.imread(path_to_file)
    image_cells = cv2.resize(image_cells, (width1, height1))

    return image_cells


def compare_contours_lists(reference_contours_list_of_dict,
                           test_contours_list_of_dict,
                           image,
                           title,
                           cnts_center_displacement,
                           cnts_area_difference_factor):
    image_copy = image.copy()
    common_contours_list_of_dict = []
    fail_csv_contours_list_of_dict = []
    fail_GT_contours_list_of_dict = []
    common_contours_list = []

    for cnt1_dict in reference_contours_list_of_dict:

        cx_1, cy_1 = cnt1_dict['cnt_center']
        cnt1_area = cnt1_dict['cnt_area']
        for cnt2_dict in test_contours_list_of_dict:

            cx_2, cy_2 = cnt2_dict['cnt_center']
            cnt2_area = cnt2_dict['cnt_area']
            if cnt1_dict['is_found'] == False and \
                    cnt2_dict['is_found'] == False and \
                    abs(cx_1 - cx_2) < cnts_center_displacement * np.sqrt(cnt1_area) and \
                    abs(cy_1 - cy_2) < cnts_center_displacement * np.sqrt(cnt1_area) and \
                    abs(cnt1_area - cnt2_area) < cnts_area_difference_factor * cnt1_area:
                cnt1_dict['is_found'] = True
                cnt2_dict['is_found'] = True

                common_contours_list_of_dict.append(cnt1_dict)
                common_contours_list.append(cnt1_dict['cnt'])

    for cnt1_dict in reference_contours_list_of_dict:
        if cnt1_dict['is_found'] == False:
            fail_GT_contours_list_of_dict.append(cnt1_dict)

    for cnt2_dict in test_contours_list_of_dict:
        if cnt2_dict['is_found'] == False:
            fail_csv_contours_list_of_dict.append(cnt2_dict)

    # draw_contours_on_image(contours=common_contours_list, color='G', image=image_copy, title=title, isFilled=True)

    return common_contours_list_of_dict, fail_GT_contours_list_of_dict, fail_csv_contours_list_of_dict


def calculate_area_of_contours(contours_list_of_dict):
    area = 0
    for cnt_dict in contours_list_of_dict:
        cnt_area = cnt_dict['cnt_area']
        area += cnt_area

    return area


def make_csv_from_cnts_list(path_to_save_csv_file, cnts_list):
    with open(path_to_save_csv_file, 'a') as file:
        for i in range(len(cnts_list)):
            cnt = cnts_list[i]
            number_of_points = len(cnt)
            # print('number_of_points=',number_of_points)
            print('cnt=', cnt)
            for j in range(number_of_points):
                print('cnt[' + str(j) + '][0][0]=', cnt[j][0][0])
                print('cnt[' + str(j) + '][0][1]=', cnt[j][0][1])
                # print('cnt['+str(j)+'][1]=',cnt[j][1])
                file.write(str(cnt[j][0][0]) + ', ' + str(cnt[j][0][1]) + ', ')
            file.write('\n')




def generate_matplotlib_plots(cnts_center_displacement_list,
                              cnts_area_difference_factor_list,

                              common_cnt_number_list,
                              fail_GT_cnt_number_list,
                              fail_method_cnt_number_list,
                              accuracy_cnt_number_list,

                              common_cnt_area_list,
                              fail_GT_cnt_area_list,
                              fail_method_cnt_area_list,
                              accuracy_cnt_area_list,
                              criteria):
    if criteria == CRITERIA_TYPE_CENTER_COORDINATES:
        # plt.subplot(121)
        common_cnt_plot, = plt.plot(cnts_center_displacement_list, common_cnt_number_list)
        plt.xlabel('displacement of centers factor')
        #plt.ylabel('Number of common contours')
        plt.ylabel('Number of contours')
        # plt.show()

        fail_GT_cnt_plot, = plt.plot(cnts_center_displacement_list, fail_GT_cnt_number_list)
        plt.xlabel('displacement of centers factor')
        #plt.ylabel('Number of fail GT contours')
        plt.ylabel('Number of contours')
        # plt.show()

        fail_method_cnt_plot, = plt.plot(cnts_center_displacement_list, fail_method_cnt_number_list)
        plt.xlabel('displacement of centers factor')
        #plt.ylabel('Number of fail method contours')
        plt.ylabel('Number of contours')
        # plt.show()
        plt.legend((common_cnt_plot, fail_GT_cnt_plot, fail_method_cnt_plot), ('common', 'fail GT', 'fail method'))
        PATH_TO_SAVE_PLOT = PATH_TO_FOLDER_OUTPUT_MATPLOTLIB_PLOTS + 'CNTs_NUMBER_' + FILE_NAME_IMAGE_INITIAL[
                                                                                      :-4] + '_' + file_type + '_' + criteria + '.png'
        plt.savefig(PATH_TO_SAVE_PLOT)
        plt.show()

        # plt.subplot(122)
        plt.plot(cnts_center_displacement_list, accuracy_cnt_number_list)
        plt.xlabel('displacement of centers factor')
        plt.ylabel('Accuracy of method by cnt number')

        PATH_TO_SAVE_PLOT = PATH_TO_FOLDER_OUTPUT_MATPLOTLIB_PLOTS + 'ACCURACY_CNT_NUMBER_' + FILE_NAME_IMAGE_INITIAL[
                                                                                              :-4] + '_' + file_type + '_' + criteria + '.png'
        plt.savefig(PATH_TO_SAVE_PLOT)
        plt.show()

        # plt.subplot(121)
        common_cnt_area_plot, = plt.plot(cnts_center_displacement_list, common_cnt_area_list)
        plt.xlabel('displacement of centers factor')
        #plt.ylabel('Area of common contours')
        plt.ylabel('Area of contours')
        # plt.show()

        fail_GT_cnt_area_plot, = plt.plot(cnts_center_displacement_list, fail_GT_cnt_area_list)
        plt.xlabel('displacement of centers factor')
        #plt.ylabel('Area of fail GT contours')
        plt.ylabel('Area of contours')
        # plt.show()

        fail_method_cnt_area_plot, = plt.plot(cnts_center_displacement_list, fail_method_cnt_area_list)
        plt.xlabel('displacement of centers factor')
        #plt.ylabel('Area of fail method contours')
        plt.ylabel('Area of contours')
        # plt.show()
        plt.legend((common_cnt_area_plot, fail_GT_cnt_area_plot, fail_method_cnt_area_plot),
                   ('common', 'fail GT', 'fail method'))
        PATH_TO_SAVE_PLOT = PATH_TO_FOLDER_OUTPUT_MATPLOTLIB_PLOTS + 'CNTs_AREA_' + FILE_NAME_IMAGE_INITIAL[
                                                                                    :-4] + '_' + file_type + '_' + criteria + '.png'
        plt.savefig(PATH_TO_SAVE_PLOT)
        plt.show()

        # plt.subplot(122)
        plt.plot(cnts_center_displacement_list, accuracy_cnt_area_list)
        plt.xlabel('displacement of centers factor')
        plt.ylabel('Accuracy of method by cnt area')

        PATH_TO_SAVE_PLOT = PATH_TO_FOLDER_OUTPUT_MATPLOTLIB_PLOTS + 'ACCURACY_CNT_AREA_' + FILE_NAME_IMAGE_INITIAL[
                                                                                            :-4] + '_' + file_type + '_' + criteria + '.png'
        plt.savefig(PATH_TO_SAVE_PLOT)
        plt.show()
    elif criteria == CRITERIA_TYPE_AREA:
        # plt.subplot(121)
        common_cnt_plot, = plt.plot(cnts_area_difference_factor_list, common_cnt_number_list)
        plt.xlabel('difference of contours area')
        #plt.ylabel('Number of common contours')
        plt.ylabel('Number of contours')
        # plt.show()

        fail_GT_cnt_plot, = plt.plot(cnts_area_difference_factor_list, fail_GT_cnt_number_list)
        plt.xlabel('difference of contours area')
        #plt.ylabel('Number of fail GT contours')
        plt.ylabel('Number of contours')
        # plt.show()

        fail_method_cnt_plot, = plt.plot(cnts_area_difference_factor_list, fail_method_cnt_number_list)
        plt.xlabel('difference of contours area')
        #plt.ylabel('Number of fail method contours')
        plt.ylabel('Number of contours')
        # plt.show()
        plt.legend((common_cnt_plot, fail_GT_cnt_plot, fail_method_cnt_plot), ('common', 'fail GT', 'fail method'))
        PATH_TO_SAVE_PLOT = PATH_TO_FOLDER_OUTPUT_MATPLOTLIB_PLOTS + 'CNTs_NUMBER_' + FILE_NAME_IMAGE_INITIAL[
                                                                                      :-4] + '_' + file_type + '_' + criteria + '.png'
        plt.savefig(PATH_TO_SAVE_PLOT)
        plt.show()

        # plt.subplot(122)
        plt.plot(cnts_area_difference_factor_list, accuracy_cnt_number_list)
        plt.xlabel('difference of contours area')
        plt.ylabel('Accuracy of method by cnt number')

        PATH_TO_SAVE_PLOT = PATH_TO_FOLDER_OUTPUT_MATPLOTLIB_PLOTS + 'ACCURACY_CNT_NUMBER_' + FILE_NAME_IMAGE_INITIAL[
                                                                                              :-4] + '_' + file_type + '_' + criteria + '.png'
        plt.savefig(PATH_TO_SAVE_PLOT)
        plt.show()

        # plt.subplot(121)
        common_cnt_area_plot, = plt.plot(cnts_area_difference_factor_list, common_cnt_area_list)
        plt.xlabel('difference of contours area')
        #plt.ylabel('Area of common contours')
        plt.ylabel('Area of contours')
        # plt.show()

        fail_GT_cnt_area_plot, = plt.plot(cnts_area_difference_factor_list, fail_GT_cnt_area_list)
        plt.xlabel('difference of contours area')
        #plt.ylabel('Area of fail GT contours')
        plt.ylabel('Area of contours')
        # plt.show()

        fail_method_cnt_area_plot, = plt.plot(cnts_area_difference_factor_list, fail_method_cnt_area_list)
        plt.xlabel('difference of contours area')
        #plt.ylabel('Area of fail method contours')
        plt.ylabel('Area of contours')
        # plt.show()
        plt.legend((common_cnt_area_plot, fail_GT_cnt_area_plot, fail_method_cnt_area_plot),
                   ('common', 'fail GT', 'fail method'))
        PATH_TO_SAVE_PLOT = PATH_TO_FOLDER_OUTPUT_MATPLOTLIB_PLOTS + 'CNTs_AREA_' + FILE_NAME_IMAGE_INITIAL[
                                                                                    :-4] + '_' + file_type + '_' + criteria + '.png'
        plt.savefig(PATH_TO_SAVE_PLOT)
        plt.show()

        # plt.subplot(122)
        plt.plot(cnts_area_difference_factor_list, accuracy_cnt_area_list)
        plt.xlabel('difference of contours area')
        plt.ylabel('Accuracy of method by cnt area')

        PATH_TO_SAVE_PLOT = PATH_TO_FOLDER_OUTPUT_MATPLOTLIB_PLOTS + 'ACCURACY_CNT_AREA_' + FILE_NAME_IMAGE_INITIAL[
                                                                                            :-4] + '_' + file_type + '_' + criteria + '.png'
        plt.savefig(PATH_TO_SAVE_PLOT)
        plt.show()
    else:
        print('ERROR! WRONG CRITERIA!')


def main():
    CRITERIA_TYPE_CENTER_COORDINATES = 'center'
    CRITERIA_TYPE_AREA = 'area'

    # COMARISON_CRITERIA
    cnts_center_displacement = 0.3
    cnts_area_difference_factor = 0.3
    #criteria = CRITERIA_TYPE_CENTER_COORDINATES

    cnts_center_displacement_list = []
    cnts_area_difference_factor_list = []

    common_cnt_number_list = []
    fail_GT_cnt_number_list = []
    fail_method_cnt_number_list = []
    accuracy_cnt_number_list = []

    common_cnt_area_list = []
    fail_GT_cnt_area_list = []
    fail_method_cnt_area_list = []
    accuracy_cnt_area_list = []

    for i in range(ITERATIONS_NUMBER):

        if criteria == CRITERIA_TYPE_CENTER_COORDINATES:
            cnts_center_displacement = MAX_CENTER_DISPLACEMENT / ITERATIONS_NUMBER * i
            cnts_area_difference_factor = 0.3
        elif criteria == CRITERIA_TYPE_AREA:
            cnts_center_displacement = 0.3
            cnts_area_difference_factor = MAX_AREA_DIFFERENCE / ITERATIONS_NUMBER * i
        else:
            print('ERROR! WRONG CRITERIA!')

        image_cells = get_resized_image_for_validation(path_to_file=PATH_TO_IMAGE_GROUND_TRUTH)

        image_cells_copy = image_cells.copy()

        contours_from_file = generate_contours_list_from_file(path_to_file=PATH_TO_FILE_CSV,
                                                              file_type=FILE_TYPE_CSV)

        contours_from_image_GROUND_TRUTH = generate_contours_list_from_GROUND_TRUTH_file(image=image_cells,
                                                                                         title='Ellipses_' + FILE_NAME_IMAGE_INITIAL[
                                                                                                             :-4])

        # ----------------------------------VALIDATION--------------------------------------------
        list_of_file_cnt_data_dict = get_list_of_cnt_data_dict(contours=contours_from_file, color='Y')

        list_of_GT_1_cnt_data_dict = get_list_of_cnt_data_dict(contours=contours_from_image_GROUND_TRUTH, color='G')

        common_contours_list_of_dict_1, fail_GT_contours_list_of_dict_1, fail_csv_contours_list_of_dict_1 = compare_contours_lists(
            reference_contours_list_of_dict=list_of_GT_1_cnt_data_dict,
            test_contours_list_of_dict=list_of_file_cnt_data_dict,
            image=image_cells_copy,
            title='validation_' + FILE_NAME_IMAGE_INITIAL[:-4],
            cnts_center_displacement=cnts_center_displacement,
            cnts_area_difference_factor=cnts_area_difference_factor)

        print('len(common_contours_list_of_dict_1)=', len(common_contours_list_of_dict_1))
        print('len(fail_GT_contours_list_of_dict_1)=', len(fail_GT_contours_list_of_dict_1))
        print('len(fail_csv_contours_list_of_dict_1)=', len(fail_csv_contours_list_of_dict_1))

        # ----------------------------------CALCULATE_CNT_NUMBER_PROBABILITY--------------------------------------------
        print('==================================================================')
        Number_GT_contours_1 = len(list_of_GT_1_cnt_data_dict)
        Number_common_1 = len(common_contours_list_of_dict_1)
        accuracy_cnt_number = Number_common_1 / Number_GT_contours_1
        print('Number_common_1/Number_GT_contours_1=', accuracy_cnt_number)

        # ----------------------------------CALCULATE_AREA_PROBABILITY--------------------------------------------
        print('==================================================================')
        S_GT_1 = calculate_area_of_contours(list_of_GT_1_cnt_data_dict)
        print('calculate_area_of_contours(list_of_GT_1_cnt_data_dict)=', S_GT_1)
        # print('calculate_area_of_contours(list_of_csv_1_cnt_data_dict)=', calculate_area_of_contours(list_of_csv_1_cnt_data_dict))
        print()
        S_common_1 = calculate_area_of_contours(common_contours_list_of_dict_1)
        fail_GT_cnt_area = calculate_area_of_contours(fail_GT_contours_list_of_dict_1)
        fail_method_cnt_area = calculate_area_of_contours(fail_csv_contours_list_of_dict_1)
        print('calculate_area_of_contours(common_contours_list_of_dict_1)=', S_common_1)
        print('calculate_area_of_contours(fail_GT_contours_list_of_dict_1)=', fail_GT_cnt_area)
        print('calculate_area_of_contours(fail_csv_contours_list_of_dict_1)=', fail_method_cnt_area)
        print()

        accuracy_cnt_area = S_common_1 / S_GT_1
        print('S(common_1)/S(GT_1)=', accuracy_cnt_area)

        cnts_center_displacement_list.append(cnts_center_displacement)
        cnts_area_difference_factor_list.append(cnts_area_difference_factor)

        common_cnt_number_list.append(len(common_contours_list_of_dict_1))
        fail_GT_cnt_number_list.append(len(fail_GT_contours_list_of_dict_1))
        fail_method_cnt_number_list.append(len(fail_csv_contours_list_of_dict_1))
        accuracy_cnt_number_list.append(accuracy_cnt_number)

        common_cnt_area_list.append(S_common_1)
        fail_GT_cnt_area_list.append(fail_GT_cnt_area)
        fail_method_cnt_area_list.append(fail_method_cnt_area)
        accuracy_cnt_area_list.append(accuracy_cnt_area)

    # print('cnts_center_displacement_list=', cnts_center_displacement_list)
    # print('cnts_area_difference_factor_list=', cnts_area_difference_factor_list)
    #
    # print('common_cnt_number_list=', common_cnt_number_list)
    # print('fail_GT_cnt_number_list=', fail_GT_cnt_number_list)
    # print('fail_method_cnt_number_list=', fail_method_cnt_number_list)
    # print('accuracy_cnt_number_list=', accuracy_cnt_number_list)
    #
    # print('common_cnt_area_list=', common_cnt_area_list)
    # print('fail_GT_cnt_area_list=', fail_GT_cnt_area_list)
    # print('fail_method_cnt_area_list=', fail_method_cnt_area_list)
    # print('accuracy_cnt_area_list=', accuracy_cnt_area_list)

    generate_matplotlib_plots(cnts_center_displacement_list=cnts_center_displacement_list,
                              cnts_area_difference_factor_list=cnts_area_difference_factor_list,

                              common_cnt_number_list=common_cnt_number_list,
                              fail_GT_cnt_number_list=fail_GT_cnt_number_list,
                              fail_method_cnt_number_list=fail_method_cnt_number_list,
                              accuracy_cnt_number_list=accuracy_cnt_number_list,

                              common_cnt_area_list=common_cnt_area_list,
                              fail_GT_cnt_area_list=fail_GT_cnt_area_list,
                              fail_method_cnt_area_list=fail_method_cnt_area_list,
                              accuracy_cnt_area_list=accuracy_cnt_area_list,
                              criteria=criteria)


if __name__ == "__main__":
    main()
