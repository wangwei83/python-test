from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import cv2

# 初始化 COCO API
dataDir = '/path/to/coco'
dataType = 'train2017'
annFile = f'{dataDir}/annotations/instances_{dataType}.json'
coco = COCO(annFile)

# 加载类别
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

# 加载图像
imgIds = coco.getImgIds()
img = coco.loadImgs(imgIds[0])[0]

# 显示图像
I = cv2.imread(f'{dataDir}/{dataType}/{img["file_name"]}')
plt.axis('off')
plt.imshow(I)
plt.show()
