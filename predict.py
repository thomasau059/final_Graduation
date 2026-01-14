import os
import sys
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# ✅ Cấu hình font cho tiếng Trung
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # Sửa lỗi dấu trừ

print("🍉 西瓜批量預測系統 🍉")
print("="*50)

class WatermelonPredictor:
    def __init__(self, model_path='./models/watermelon_model.h5'):
        print("📄 載入模型中...")
        try:
            self.model = load_model(model_path)
            print(f"✅ 模型載入成功: {model_path}")
        except Exception as e:
            print(f"❌ 模型載入錯誤: {e}")
            sys.exit(1)
        
        self.class_names = ['good', 'not_good', 'others']
        self.class_indices = {i: name for i, name in enumerate(self.class_names)}
        self.img_height = 224
        self.img_width = 224
        self.confidence_threshold = 0.60
        
        self.results_dir = f"predict_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"📁 結果將保存到: {self.results_dir}")
        print(f"🏷️ 類別: {self.class_names}")
        print(f"🎯 信心度閾值: {self.confidence_threshold}")
        print("="*50)
    
    def preprocess_image(self, image_path):
        try:
            img = load_img(image_path, target_size=(self.img_height, self.img_width))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            return img_array, True
        except Exception as e:
            print(f"❌ 處理錯誤 {image_path}: {e}")
            return None, False
    
    def predict_single(self, image_path):
        img_array, success = self.preprocess_image(image_path)
        if not success:
            return None
        
        predictions = self.model.predict(img_array, verbose=0)
        probs = predictions[0]
        class_idx = np.argmax(probs)
        confidence = probs[class_idx]
        class_name = self.class_indices[class_idx]
        
        result = {
            'image_path': image_path,
            'predicted_class': class_name,
            'confidence': float(confidence),
            'class_index': int(class_idx),
            'probabilities': [float(p) for p in probs],
            'is_confident': confidence >= self.confidence_threshold,
            'is_watermelon': class_name in ['good', 'not_good']
        }
        
        return result
    
    def predict_folder(self, folder_path, organize=False):
        print(f"\n🔍 預測目錄: {folder_path}")
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.jfif', '.bmp']
        image_files = []
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    full_path = os.path.join(root, file)
                    image_files.append(full_path)
        
        print(f"📸 找到 {len(image_files)} 張圖片")
        
        if len(image_files) == 0:
            print("❌ 未找到圖片!")
            return
        
        results = []
        for i, image_path in enumerate(image_files, 1):
            print(f"\r🔄 處理中 {i}/{len(image_files)}...", end="")
            result = self.predict_single(image_path)
            if result:
                results.append(result)
        
        print(f"\n✅ 成功處理 {len(results)} 張圖片")
        self.generate_report(results, organize)
        
        return results
    
    def generate_report(self, results, organize=False):
        print("\n📊 生成報告中...")
        
        df = pd.DataFrame(results)
        df['filename'] = df['image_path'].apply(os.path.basename)
        df['folder'] = df['image_path'].apply(os.path.dirname)
        
        csv_path = os.path.join(self.results_dir, 'predictions.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ CSV 已保存: {csv_path}")
        
        summary = df['predicted_class'].value_counts().to_dict()
        
        with open(os.path.join(self.results_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("西瓜預測報告\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"總圖片數: {len(df)}\n")
            f.write(f"預測時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("類別分佈:\n")
            for class_name in self.class_names:
                count = summary.get(class_name, 0)
                percentage = (count / len(df)) * 100 if len(df) > 0 else 0
                f.write(f"  {class_name}: {count} ({percentage:.1f}%)\n")
            
            f.write(f"\n信心度統計:\n")
            f.write(f"  平均信心度: {df['confidence'].mean():.2%}\n")
            f.write(f"  最低信心度: {df['confidence'].min():.2%}\n")
            f.write(f"  最高信心度: {df['confidence'].max():.2%}\n")
            
            confident_count = df['is_confident'].sum()
            f.write(f"\n高信心度預測 (≥{self.confidence_threshold:.0%}): {confident_count}/{len(df)} ({confident_count/len(df):.1%})\n")
            
            watermelon_count = df['is_watermelon'].sum()
            f.write(f"西瓜圖片: {watermelon_count}/{len(df)} ({watermelon_count/len(df):.1%})\n")
        
        print(f"✅ 報告已保存: {self.results_dir}/summary.txt")
        self.create_visualizations(df)
        
        if organize:
            self.organize_images(df)
    
    def create_visualizations(self, df):
        print("📈 生成圖表中...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('西瓜預測分析', fontsize=16, fontweight='bold')
        
        # 1. 類別分佈
        ax1 = axes[0, 0]
        class_counts = df['predicted_class'].value_counts()
        colors = ['#4CAF50', '#f44336', '#ff9800']
        class_counts.plot(kind='bar', ax=ax1, color=colors, edgecolor='black')
        ax1.set_title('類別分佈', fontweight='bold', fontsize=12)
        ax1.set_xlabel('類別', fontsize=10)
        ax1.set_ylabel('數量', fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        
        for i, (class_name, count) in enumerate(class_counts.items()):
            ax1.text(i, count + 0.5, str(count), ha='center', va='bottom', fontweight='bold')
        
        # 2. 信心度分佈
        ax2 = axes[0, 1]
        df['confidence'].hist(bins=20, ax=ax2, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.axvline(self.confidence_threshold, color='red', linestyle='--', 
                   label=f'閾值 ({self.confidence_threshold:.0%})')
        ax2.set_title('信心度分佈', fontweight='bold', fontsize=12)
        ax2.set_xlabel('信心度', fontsize=10)
        ax2.set_ylabel('頻率', fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)
        
        # 3. 類別信心度散點圖
        ax3 = axes[1, 0]
        for i, class_name in enumerate(self.class_names):
            class_data = df[df['predicted_class'] == class_name]
            if len(class_data) > 0:
                ax3.scatter([i] * len(class_data), class_data['confidence'], 
                          alpha=0.6, label=class_name, s=50)
        ax3.set_title('各類別信心度', fontweight='bold', fontsize=12)
        ax3.set_xlabel('類別', fontsize=10)
        ax3.set_ylabel('信心度', fontsize=10)
        ax3.set_xticks(range(len(self.class_names)))
        ax3.set_xticklabels(self.class_names, rotation=45)
        ax3.axhline(self.confidence_threshold, color='red', linestyle='--', alpha=0.5)
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3)
        
        # 4. 信心度級別餅圖
        ax4 = axes[1, 1]
        confident_count = df['is_confident'].sum()
        uncertain_count = len(df) - confident_count
        sizes = [confident_count, uncertain_count]
        labels = [f'高信心度\n({confident_count})', f'低信心度\n({uncertain_count})']
        colors_pie = ['#4CAF50', '#ff9800']
        
        ax4.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
               startangle=90, wedgeprops={'edgecolor': 'black', 'linewidth': 2},
               textprops={'fontsize': 10})
        ax4.set_title('信心度級別分佈', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'analysis_charts.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 圖表已保存: {self.results_dir}/analysis_charts.png")
    
    def organize_images(self, df):
        print("\n🗂️ 整理圖片到分類目錄...")
        
        organize_dir = os.path.join(self.results_dir, 'organized')
        os.makedirs(organize_dir, exist_ok=True)
        
        for class_name in self.class_names:
            class_dir = os.path.join(organize_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
        
        uncertain_dir = os.path.join(organize_dir, 'uncertain')
        os.makedirs(uncertain_dir, exist_ok=True)
        
        for _, row in df.iterrows():
            src_path = row['image_path']
            filename = row['filename']
            
            if row['is_confident']:
                dest_dir = os.path.join(organize_dir, row['predicted_class'])
            else:
                dest_dir = uncertain_dir
            
            dest_path = os.path.join(dest_dir, filename)
            
            try:
                shutil.copy2(src_path, dest_path)
            except Exception as e:
                print(f"❌ 複製錯誤 {filename}: {e}")
        
        print(f"✅ 圖片已整理到: {organize_dir}")

def main():
    print("🍉 西瓜分類預測系統")
    print("="*50)
    
    predictor = WatermelonPredictor()
    
    while True:
        print("\n" + "="*50)
        print("選單:")
        print("1. 預測單張圖片")
        print("2. 預測整個目錄")
        print("3. 預測並整理圖片")
        print("4. 退出")
        print("="*50)
        
        choice = input("請選擇 (1-4): ").strip()
        
        if choice == '1':
            image_path = input("輸入圖片路徑: ").strip()
            if os.path.exists(image_path):
                result = predictor.predict_single(image_path)
                if result:
                    print("\n" + "="*50)
                    print("預測結果:")
                    print("="*50)
                    print(f"圖片: {os.path.basename(image_path)}")
                    print(f"預測類別: {result['predicted_class'].upper()}")
                    print(f"信心度: {result['confidence']:.2%}")
                    print(f"高信心度: {'✅' if result['is_confident'] else '❌'}")
                    print(f"是西瓜: {'✅' if result['is_watermelon'] else '❌'}")
                    print("\n機率分佈:")
                    for i, prob in enumerate(result['probabilities']):
                        class_name = predictor.class_indices[i]
                        print(f"  {class_name}: {prob:.2%}")
                    print("="*50)
            else:
                print("❌ 文件不存在!")
        
        elif choice == '2':
            folder_path = input("輸入目錄路徑: ").strip()
            if os.path.exists(folder_path):
                predictor.predict_folder(folder_path, organize=False)
            else:
                print("❌ 目錄不存在!")
        
        elif choice == '3':
            folder_path = input("輸入目錄路徑: ").strip()
            if os.path.exists(folder_path):
                predictor.predict_folder(folder_path, organize=True)
            else:
                print("❌ 目錄不存在!")
        
        elif choice == '4':
            print("👋 再見!")
            break
        
        else:
            print("❌ 無效選擇! 請輸入 1-4")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='西瓜圖片分類器')
    parser.add_argument('--image', type=str, help='單張圖片路徑')
    parser.add_argument('--folder', type=str, help='圖片目錄路徑')
    parser.add_argument('--organize', action='store_true', help='預測後整理圖片')
    parser.add_argument('--model', type=str, default='./models/watermelon_model.h5', 
                       help='模型文件路徑')
    
    args = parser.parse_args()
    
    if args.image or args.folder:
        predictor = WatermelonPredictor(args.model)
        
        if args.image:
            if os.path.exists(args.image):
                result = predictor.predict_single(args.image)
                if result:
                    print(f"\n結果: {result['predicted_class']} ({result['confidence']:.2%})")
            else:
                print(f"❌ 圖片未找到: {args.image}")
        
        if args.folder:
            if os.path.exists(args.folder):
                predictor.predict_folder(args.folder, organize=args.organize)
            else:
                print(f"❌ 目錄未找到: {args.folder}")
    else:
        main()