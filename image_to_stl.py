import cv2
import numpy as np
from PIL import Image
import trimesh
from stl import mesh
import matplotlib.pyplot as plt
from rembg import remove
import io

class ImageTo3D:
    def __init__(self, max_height=10.0, base_thickness=2.0):
        self.max_height = max_height
        self.base_thickness = base_thickness
    
    def remove_background(self, image_path, method='ai'):
        if method == 'ai':
            with open(image_path, 'rb') as input_file:
                input_data = input_file.read()
            output_data = remove(input_data)
            
            img_pil = Image.open(io.BytesIO(output_data)).convert('RGBA')
            img_array = np.array(img_pil)
            
            alpha = img_array[:, :, 3]
            mask = (alpha > 128).astype(np.uint8) * 255
            
            rgb_img = cv2.cvtColor(img_array[:, :, :3], cv2.COLOR_RGB2BGR)
            
            return rgb_img, mask
        else:
            return self._traditional_background_removal(image_path)
    
    def _traditional_background_removal(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Görüntü yüklenemedi: {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        markers = cv2.watershed(img, markers)
        
        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[markers > 1] = 255
        
        return img, mask
    
    def create_depth_map(self, image_path, method='gradient', remove_bg=True):
        if remove_bg:
            img, mask = self.remove_background(image_path, 'ai')
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            mask = np.ones(img_gray.shape, dtype=np.uint8) * 255
            
        if img_gray is None:
            raise ValueError(f"Görüntü yüklenemedi: {image_path}")
        
        if method == 'gradient':
            depth_map = self._gradient_based_depth(img_gray)
        elif method == 'edge':
            depth_map = self._edge_based_depth(img_gray)
        else:
            depth_map = self._simple_intensity_depth(img_gray)
        
        depth_map = depth_map * (mask / 255.0)
        
        return depth_map, mask
    
    def _gradient_based_depth(self, img):
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        blurred = cv2.GaussianBlur(img.astype(np.float32), (15, 15), 0)
        intensity_depth = blurred / 255.0
        gradient_depth = 1.0 - (gradient_magnitude / np.max(gradient_magnitude))
        
        depth_map = 0.7 * intensity_depth + 0.3 * gradient_depth
        return depth_map
    
    def _edge_based_depth(self, img):
        edges = cv2.Canny(img, 50, 150)
        distance_transform = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
        depth_map = distance_transform / np.max(distance_transform)
        return depth_map
    
    def _simple_intensity_depth(self, img):
        return img.astype(np.float32) / 255.0
    
    def create_mesh_from_depth(self, depth_map, resolution_factor=1.0):
        height, width = depth_map.shape
        
        if resolution_factor != 1.0:
            new_height = int(height * resolution_factor)
            new_width = int(width * resolution_factor)
            depth_map = cv2.resize(depth_map, (new_width, new_height))
            height, width = depth_map.shape
        
        vertices = []
        faces = []
        
        for y in range(height):
            for x in range(width):
                z = depth_map[y, x] * self.max_height + self.base_thickness
                vertices.append([x, y, z])
        
        for y in range(height - 1):
            for x in range(width - 1):
                idx = y * width + x
                
                faces.append([idx, idx + 1, idx + width])
                faces.append([idx + 1, idx + width + 1, idx + width])
        
        base_z = self.base_thickness
        base_vertices = []
        base_faces = []
        
        for y in range(height):
            for x in range(width):
                base_vertices.append([x, y, base_z])
        
        for y in range(height - 1):
            for x in range(width - 1):
                idx = y * width + x + len(vertices)
                
                base_faces.append([idx, idx + width, idx + 1])
                base_faces.append([idx + 1, idx + width, idx + width + 1])
        
        all_vertices = vertices + base_vertices
        all_faces = faces + base_faces
        
        for y in range(height):
            top_idx = y * width
            bottom_idx = y * width + len(vertices)
            
            if y < height - 1:
                next_top_idx = (y + 1) * width
                next_bottom_idx = (y + 1) * width + len(vertices)
                
                all_faces.append([top_idx, bottom_idx, next_top_idx])
                all_faces.append([bottom_idx, next_bottom_idx, next_top_idx])
        
        for x in range(width):
            top_idx = x
            bottom_idx = x + len(vertices)
            
            if x < width - 1:
                next_top_idx = x + 1
                next_bottom_idx = x + 1 + len(vertices)
                
                all_faces.append([top_idx, next_top_idx, bottom_idx])
                all_faces.append([next_top_idx, next_bottom_idx, bottom_idx])
        
        return np.array(all_vertices), np.array(all_faces)
    
    def save_as_stl(self, vertices, faces, output_path):
        stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        
        for i, face in enumerate(faces):
            for j in range(3):
                stl_mesh.vectors[i][j] = vertices[face[j]]
        
        stl_mesh.save(output_path)
        print(f"STL dosyası kaydedildi: {output_path}")
    
    def process_image_to_stl(self, image_path, output_path, method='gradient', resolution_factor=0.5, remove_bg=True):
        print(f"Görüntü işleniyor: {image_path}")
        
        if remove_bg:
            print("Arkaplan kaldırılıyor...")
        
        depth_map, mask = self.create_depth_map(image_path, method, remove_bg)
        print("Depth map oluşturuldu")
        
        vertices, faces = self.create_mesh_from_depth(depth_map, resolution_factor)
        print(f"3D mesh oluşturuldu - {len(vertices)} vertex, {len(faces)} face")
        
        self.save_as_stl(vertices, faces, output_path)
        
        return depth_map, mask

def main():
    converter = ImageTo3D(max_height=10.0, base_thickness=2.0)
    
    image_path = input("2D görüntü yolu girin: ").strip()
    output_path = input("Çıktı STL dosya yolu girin (örn: output.stl): ").strip()
    
    if not output_path.endswith('.stl'):
        output_path += '.stl'
    
    remove_bg = input("Arkaplanı kaldır? (y/n, önerilen y): ").strip().lower() != 'n'
    
    print("\nDepth map yöntemini seçin:")
    print("1. Gradient tabanlı (önerilen)")
    print("2. Edge tabanlı")
    print("3. Basit intensite")
    
    method_choice = input("Seçiminiz (1-3): ").strip()
    method_map = {'1': 'gradient', '2': 'edge', '3': 'intensity'}
    method = method_map.get(method_choice, 'gradient')
    
    resolution = float(input("Çözünürlük faktörü (0.1-1.0, önerilen 0.5): ") or "0.5")
    
    try:
        depth_map, mask = converter.process_image_to_stl(image_path, output_path, method, resolution, remove_bg)
        
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        original = cv2.imread(image_path)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title('Orijinal Görüntü')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title('Nesne Maskesi')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(depth_map, cmap='viridis')
        plt.title('Depth Map')
        plt.colorbar()
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n✅ İşlem tamamlandı!")
        print(f"STL dosyası: {output_path}")
        print("3D printer'da yazdırabilirsiniz.")
        
    except Exception as e:
        print(f"❌ Hata: {e}")

if __name__ == "__main__":
    main()