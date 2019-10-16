import numpy
import cv2

cnt1 = numpy.array([[1, 1], [10, 50], [50, 50]], dtype=numpy.int32)
cnt2 = numpy.array([[99, 99], [99, 60], [60, 99]], dtype=numpy.int32)
cnt3 = numpy.empty((1,2), dtype=numpy.int32)

cnt1=numpy.append(cnt1,[[22,33]],axis=0)
cnt3[0]=[100,200]
cnt3=numpy.append(cnt3,[[22,33]],axis=0)
print(cnt1)
print(cnt1[0])
print(cnt1[1])
print(cnt1[1][0])
print('=================')
print(cnt1[0][0],', ',cnt1[0][1])
print(cnt1[1][0],', ',cnt1[1][1])
print(cnt1[2][0],', ',cnt1[2][1])
print(cnt1[3][0],', ',cnt1[3][1])

print('-----------------')
print('cnt3=',cnt3)

contours = [cnt1, cnt2]

drawing = numpy.zeros([100, 100], numpy.uint8)
for cnt in contours:
    cv2.drawContours(drawing, [cnt], 0, (255, 255, 255), 2)

cv2.imshow('output', drawing)
cv2.waitKey(0)
