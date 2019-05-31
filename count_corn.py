# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 19:18:14 2019

@author: Jefferson
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

# Read image
#img = cv2.imread("inference_10m_v2_1.png", cv2.IMREAD_GRAYSCALE)
img = cv2.imread("21.png")

kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# Setup SimpleBlobDetector parameters.
# O algoritmo do SimpleBlobDetector é controlado pelos parâmetros de Thresholding, onde ele converte as imagens de origem em várias imagens binárias, limitando a imagem de origem com limites começando em minThreshold. Esses limites são incrementados por thresholdStep até maxThreshold. Portanto, o primeiro limite é minThreshold, o segundo é minThreshold + thresholdStep, o terceiro é minThreshold + 2 x thresholdStep e assim por diante. Grouping: Em cada imagem binária, os pixels brancos conectados são agrupados. Vamos chamar esses de blobs binários. Merging: Os centros dos blobs binários nas imagens binárias são calculados, e os blobs localizados mais próximos do que minDistBetweenBlobs são mesclados. Cálculo de Centro e Raio: os centros e os raios dos novos blobs fundidos são calculados e o valor resultante é retornado
params = cv2.SimpleBlobDetector_Params()

# Filter by Area.
# No nosso caso iremos filtrar somente por área e não por cor, portanto filterbyArea = True. O valor da área é dado em pixels, exemplo: minArea = 100 irá filtrar todos os blobs que possuem menos de 100 pixels.
params.filterByColor = True
params.blobColor = 255

params.filterByArea = True
params.minArea = 50

# Filter by Shape
# Filtrando por forma existem 3 parâmetros diferentes: Circularity mede o quão perto de um círculo está o blob. Por exemplo, um hexágono regular tem maior circularidade do que um quadrado. Para filtrar por circularidade, definir filterByCircularity = 1. Em seguida, é necessário definir os valores apropriados para minCircularity e maxCircularity. Circularidade é definida como 4piArea/(perímetro)², ou seja, um círculo possui uma circularidade de 1, a circularidade de um quadrado é 0.785 e assim por diante.
params.filterByCircularity = False

# A convexidade é definida como a (Área do Blob / Área do seu casco convexo). Agora, o Convexo de Hull de uma forma é a forma convexa mais apertada que envolve completamente a forma. Para filtrar por convexidade, definir filterByConvexity = 1, seguido de configuração 0 ≤ minConvexity ≤ 1 e maxConvexity (≤ 1)
params.filterByConvexity = False
params.minConvexity = 0.5


# Mede quão alongada é a forma. Por exemplo, para um círculo, esse valor é 1, para uma elipse é entre 0 e 1 e para uma linha é 0. Para filtrar por taxa de inércia, defina filterByInertia = 1 e defina 0 ≤ minInertiaRatio ≤ 1 e maxInertiaRatio (≤ 1 ) adequadamente.
params.filterByInertia = False

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
   detector = cv2.SimpleBlobDetector(params)
else :
   detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(opening)

# Draw detected blobs as red circles.
im_with_keypoints = cv2.drawKeypoints(opening, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite("corns_detected_21.png", im_with_keypoints)
print("Quantidade de pés de milho detectados: " + str(len(keypoints)))