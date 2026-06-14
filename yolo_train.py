from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")  # yolo11n 已经是最小的模型了
    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=512,
        batch=-1,

        # 模型热身
        lr0=0.001,  # 降低初始学习率，小数据集更稳定
        lrf=0.01,  # 最终学习率倍率
        warmup_epochs=2,  # 减少热身轮数，数据集小不需要太长热身

        # 图像增强（小数据集核心策略）
        mosaic=0.5,  # 启用马赛克增强，但降低概率
        mixup=0.3,  # 启用混合增强
        copy_paste=0.3,  # 启用复制粘贴增强
        fliplr=0.5,  # 50%概率水平翻转
        degrees=5,  # 减少旋转角度
        translate=0.1,  # 平移
        scale=0.3,  # 缩放范围缩小
        shear=5,  # 剪切变换，增加多样性
        perspective=0.0,  # 透视变换，小数据集建议关闭
        hsv_h=0.01,
        hsv_s=0.2,  # 降低饱和度变化范围
        hsv_v=0.2,  # 降低明度变化范围
        erasing=0.3,  # 随机擦除，增强泛化能力
        crop_fraction=0.8,  # 裁剪比例

        # 正则化（强化正则化防止过拟合）
        dropout=0.1,  # 增加dropout比例
        weight_decay=0.0005,  # 增加权重衰减

        # 训练策略优化
        close_mosaic=10,  # 最后10个epoch关闭mosaic增强
        patience=20,  # 早停耐心值，验证集不提升20轮就停止
        save_period=10,  # 每10轮保存一次模型

        device=0,
        cache="ram",
        workers=1,
        project="test2train",
        name="true_number_small",
    )
