import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import json
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False 
print(" 西瓜分類模型訓練系統 ")
print("="*50)

# 配置參數
train_data_path = './data/train'
validation_data_path = './data/validation'
model_save_path = './models/watermelon_model.h5'

img_width, img_height = 224, 224
batch_size = 32
epochs = 30
learning_rate = 1e-4
num_classes = 3  # good, not_good, others

# 創建模目錄
os.makedirs('./models', exist_ok=True)

print(" 檢查數據集...")
for class_name in ['good', 'not_good', 'others']:
    train_dir = os.path.join(train_data_path, class_name)
    val_dir = os.path.join(validation_data_path, class_name)
    
    train_count = len(os.listdir(train_dir)) if os.path.exists(train_dir) else 0
    val_count = len(os.listdir(val_dir)) if os.path.exists(val_dir) else 0
    
    print(f"  {class_name}: 訓練 {train_count} 張, 驗證 {val_count} 張")

print("\n 載入 MobileNetV2 模型...")
base_model = MobileNetV2(weights='imagenet',
                         include_top=False,
                         input_shape=(img_width, img_height, 3))

# 凍結基礎層
for layer in base_model.layers:
    layer.trainable = False

# 添加自定義層
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 編譯模型
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print("\n 準備數據生成器...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

print("\n 類別索引:")
print(train_generator.class_indices)

# 保存類別索引
with open('./models/class_indices.txt', 'w') as f:
    f.write(str(train_generator.class_indices))

print("\n 設置回調函數...")
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    model_save_path,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

callbacks = [early_stopping, model_checkpoint, reduce_lr]

print(f"\n 開始訓練 {num_classes} 類別模型...")
print(f"每輪批次數: {len(train_generator)}")

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

print("\n📈 繪製訓練歷史...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 準確度圖表
axes[0].plot(history.history['accuracy'], label='訓練準確度')
axes[0].plot(history.history['val_accuracy'], label='驗證準確度')
axes[0].set_title('模型準確度')
axes[0].set_xlabel('輪次')
axes[0].set_ylabel('準確度')
axes[0].legend()
axes[0].grid(True)

# 損失圖表
axes[1].plot(history.history['loss'], label='訓練損失')
axes[1].plot(history.history['val_loss'], label='驗證損失')
axes[1].set_title('模型損失')
axes[1].set_xlabel('輪次')
axes[1].set_ylabel('損失')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('./models/training_history.png', dpi=100)
plt.show()

print("\n" + "="*50)
print("最終模型評估")
print("="*50)

val_loss, val_accuracy = model.evaluate(validation_generator)
print(f" 驗證準確度: {val_accuracy*100:.2f}%")
print(f" 驗證損失: {val_loss:.4f}")
print(f" 模型已保存: {model_save_path}")

print("\n" + "="*50)
print(" 重要信息 (用於 app.py):")
print(f"類別索引: {train_generator.class_indices}")
print("="*50)