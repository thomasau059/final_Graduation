import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

print("🍉 西瓜模型測試系統 🍉")
print("="*50)

def test_single_image(image_path, model_path='./models/watermelon_model.h5'):
    """測試單張圖片"""
    print(f"\n📸 測試圖片: {image_path}")
    
    try:
        model = load_model(model_path)
        print("✅ 模型載入成功")
    except:
        print("❌ 模型載入失敗")
        return None
    
    try:
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array, verbose=0)
        probs = predictions[0]
        class_idx = np.argmax(probs)
        
        class_names = {0: '好西瓜', 1: '不好西瓜', 2: '其他物體'}
        class_name = class_names.get(class_idx, '未知')
        
        print("="*50)
        print(f"📊 測試結果")
        print("="*50)
        print(f"圖片: {os.path.basename(image_path)}")
        print(f"結果: {class_name}")
        print(f"信心度: {probs[class_idx]*100:.2f}%")
        print("\n詳細機率:")
        print(f"  好西瓜: {probs[0]*100:.2f}%")
        print(f"  不好西瓜: {probs[1]*100:.2f}%") 
        print(f"  其他物體: {probs[2]*100:.2f}%")
        print("="*50)
        
        return class_name, probs
    
    except Exception as e:
        print(f"❌ 測試錯誤: {e}")
        return None

def test_folder(folder_path, model_path='./models/watermelon_model.h5'):
    """測試整個目錄"""
    print(f"\n📁 測試目錄: {folder_path}")
    
    try:
        model = load_model(model_path)
    except:
        print("❌ 模型載入失敗")
        return
    
    results = {'好西瓜': 0, '不好西瓜': 0, '其他物體': 0, '總數': 0}
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            
            try:
                img = load_img(image_path, target_size=(224, 224))
                img_array = img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                predictions = model.predict(img_array, verbose=0)
                class_idx = np.argmax(predictions[0])
                
                class_names = {0: '好西瓜', 1: '不好西瓜', 2: '其他物體'}
                result = class_names.get(class_idx, '未知')
                
                results[result] += 1
                results['總數'] += 1
                
                print(f"{filename}: {result}")
                
            except Exception as e:
                print(f"❌ {filename} 錯誤: {e}")
    
    print("\n" + "="*50)
    print("📈 測試總結")
    print("="*50)
    print(f"總圖片數: {results['總數']}")
    print(f"好西瓜: {results['好西瓜']} ({results['好西瓜']/results['總數']*100:.1f}%)")
    print(f"不好西瓜: {results['不好西瓜']} ({results['不好西瓜']/results['總數']*100:.1f}%)")
    print(f"其他物體: {results['其他物體']} ({results['其他物體']/results['總數']*100:.1f}%)")
    print("="*50)

if __name__ == '__main__':
    print("選擇測試模式:")
    print("1. 測試單張圖片")
    print("2. 測試目錄")
    print("3. 完整測試 (所有目錄)")
    
    choice = input("請選擇 (1-3): ").strip()
    
    if choice == '1':
        image_path = input("輸入圖片路徑: ").strip()
        if os.path.exists(image_path):
            test_single_image(image_path)
        else:
            print("❌ 圖片不存在")
    
    elif choice == '2':
        folder_path = input("輸入目錄路徑: ").strip()
        if os.path.exists(folder_path):
            test_folder(folder_path)
        else:
            print("❌ 目錄不存在")
    
    elif choice == '3':
        print("\n測試 '好西瓜' 目錄:")
        test_folder('./test-data/good') if os.path.exists('./test-data/good') else print("目錄不存在")
        
        print("\n測試 '不好西瓜' 目錄:")
        test_folder('./test-data/not_good') if os.path.exists('./test-data/not_good') else print("目錄不存在")
        
        print("\n測試 '其他物體' 目錄:")
        test_folder('./test-data/others') if os.path.exists('./test-data/others') else print("目錄不存在")
    
    else:
        print("❌ 無效選擇")