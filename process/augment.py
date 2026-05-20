import os
import cv2
import albumentations as A
from pathlib import Path
from tqdm import tqdm


def setup_directories(should_create_mask: bool, aug_images_dir: str, aug_masks_dir: str):
    if not Path(aug_images_dir).exists():
        Path(aug_images_dir).mkdir(parents=True, exist_ok=True)

    if should_create_mask:
        if not Path(aug_masks_dir).exists():
            Path(aug_masks_dir).mkdir(parents=True, exist_ok=True)
    else:
        print(f"⚠️ SHOULD_CREATE_MASK=False. Папка масок {aug_masks_dir} игнорируется.")


def get_augmentation_transform():
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.6),
        A.VerticalFlip(p=0.3),
        A.Affine(
            scale=(0.8, 1.2),
            translate_percent=(-0.1, 0.1),
            rotate=(-45, 45),
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=0.7
        ),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.12,
                contrast_limit=0.12,
                p=0.6
            ),
            A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=5, val_shift_limit=5, p=0.4),
        ], p=0.7)
    ])


def augment_dataset(should_create_mask: bool, 
                    src_images_dir: str,
                    aug_images_dir: str,
                    aug_factor: int,
                    src_masks_dir: str = None,
                    aug_masks_dir: str = None):
    setup_directories(should_create_mask, aug_images_dir, aug_masks_dir)

    transform = get_augmentation_transform()

    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(src_images_dir) if f.lower().endswith(valid_extensions)]

    print(f"Найдено {len(image_files)} исходных фото для аугментации.")
    print(f"Параметр SHOULD_CREATE_MASK: {should_create_mask}")

    if should_create_mask:
        print(f"Цель: Создать {len(image_files) * aug_factor} новых пар (фото + маска).")
    else:
        print(f"Цель: Создать {len(image_files) * aug_factor} новых фото (без масок).")

    total_generated = 0

    for filename in tqdm(image_files, desc="Аугментация"):
        name, ext = os.path.splitext(filename)

        src_img_path = os.path.join(src_images_dir, filename)

        image = cv2.imread(src_img_path)
        if image is None:
            print(f"❌ Ошибка чтения изображения: {filename}")
            continue

        mask = None
        if should_create_mask:
            src_mask_filename = f"{name}.png"
            src_mask_path = os.path.join(src_masks_dir, src_mask_filename)

            if not os.path.exists(src_mask_path):
                print(f"⚠️ Пропуск {filename}: нет соответствующей маски {src_mask_filename}")
                continue
            mask = cv2.imread(src_mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"❌ Ошибка чтения маски: {filename}")
                continue
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for i in range(aug_factor):
            try:
                aug_mask = None
                if should_create_mask:
                    augmented = transform(image=image_rgb, mask=mask)
                    aug_image_rgb = augmented['image']
                    aug_mask = augmented['mask']
                else:
                    augmented = transform(image=image_rgb)
                    aug_image_rgb = augmented['image']

                aug_image_bgr = cv2.cvtColor(aug_image_rgb, cv2.COLOR_RGB2BGR)

                new_img_name = f"{name}_aug_{i}{ext}"
                save_img_path = os.path.join(aug_images_dir, new_img_name)

                cv2.imwrite(save_img_path, aug_image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])

                if should_create_mask:
                    new_mask_name = f"{name}_aug_{i}.png"
                    save_mask_path = os.path.join(aug_masks_dir, new_mask_name)
                    cv2.imwrite(save_mask_path, aug_mask)

                total_generated += 1

            except Exception as e:
                print(f"⚠️ Ошибка при аугментации {filename} (variant {i}): {e}")

    print(f"\n✅ ГОТОВО!")
    print(f"Сгенерировано {total_generated} примеров.")
    print(f"📂 Изображения: {aug_images_dir}")
    if should_create_mask:
        print(f"📂 Маски: {aug_masks_dir}")
    else:
        print(f"⚠️ Маски не создавались.")


augment_dataset(should_create_mask=False,
                src_images_dir=r"dataset\classify\binary\tomato_binary\new_healthy",
                aug_images_dir=r"dataset\classify\binary\tomato_binary\new_healthy_aug",
                aug_factor=6)
