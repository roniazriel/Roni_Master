import pandas as pd
import matplotlib.pyplot as plt
from tab2img.converter import Tab2Img


''' 2. TabTransform'''

''' 3. Convert data to image for CNN'''
dataset = pd.read_csv("grouped_data.csv")
dataset = dataset.drop("Unnamed: 0", axis=1)
print(dataset)
print(dataset.keys())

'''Convert to image using 1 target'''
target = dataset[['Success_Rates']]
dataset.drop(['Success_Rates','Manipulability_Rates'], inplace=True, axis=1)

model = Tab2Img()
images = model.fit_transform(dataset.to_numpy(), target.to_numpy())
fig, axes = plt.subplots(10,10, figsize=(8,8))

# plot 100 images (10X10)
for i,ax in enumerate(axes.flat):
    ax.imshow(images[i])
plt.show()