python train.py
2026-02-26 20:37:27,973 - INFO - 训练设备：cpu
Files already downloaded and verified
Files already downloaded and verified
2026-02-26 20:37:30,263 - INFO - 数据集加载完成 | 训练集数量：50000 | 验证集数量：10000
D:\soft\miniconda3\envs\resnet_cifar10\Lib\site-packages\torchvision\models\_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
D:\soft\miniconda3\envs\resnet_cifar10\Lib\site-packages\torchvision\models\_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
  warnings.warn(msg)
2026-02-26 20:37:30,443 - INFO - 模型构建完成：resnet18
2026-02-26 20:37:30,444 - INFO - 
========== Epoch 1/10 ==========
2026-02-26 20:40:51,830 - INFO - Train | Loss: 1.7446 | Acc: 0.3682                                                                                                                           
2026-02-26 20:41:04,266 - INFO - Valid | Loss: 1.3527 | Acc: 0.5130                                                                                                                           
2026-02-26 20:41:04,337 - INFO - 保存最佳模型 | 验证准确率：0.5130 | 路径：./work_dirs/resnet18\best_model.pth
2026-02-26 20:41:04,338 - INFO - 
========== Epoch 2/10 ==========
2026-02-26 20:44:23,877 - INFO - Train | Loss: 1.4279 | Acc: 0.4880                                                                                                                           
2026-02-26 20:44:36,244 - INFO - Valid | Loss: 1.1966 | Acc: 0.5780                                                                                                                           
2026-02-26 20:44:36,321 - INFO - 保存最佳模型 | 验证准确率：0.5780 | 路径：./work_dirs/resnet18\best_model.pth
2026-02-26 20:44:36,321 - INFO - 
========== Epoch 3/10 ==========
2026-02-26 20:47:55,687 - INFO - Train | Loss: 1.2630 | Acc: 0.5509                                                                                                                           
2026-02-26 20:48:07,966 - INFO - Valid | Loss: 1.0464 | Acc: 0.6309                                                                                                                           
2026-02-26 20:48:08,038 - INFO - 保存最佳模型 | 验证准确率：0.6309 | 路径：./work_dirs/resnet18\best_model.pth
2026-02-26 20:48:08,038 - INFO - 
========== Epoch 4/10 ==========
2026-02-26 20:51:29,638 - INFO - Train | Loss: 1.1424 | Acc: 0.5977                                                                                                                           
2026-02-26 20:51:41,945 - INFO - Valid | Loss: 0.9880 | Acc: 0.6519                                                                                                                           
2026-02-26 20:51:42,018 - INFO - 保存最佳模型 | 验证准确率：0.6519 | 路径：./work_dirs/resnet18\best_model.pth
2026-02-26 20:51:42,019 - INFO - 
========== Epoch 5/10 ==========
2026-02-26 20:55:00,108 - INFO - Train | Loss: 1.0553 | Acc: 0.6284                                                                                                                           
2026-02-26 20:55:12,421 - INFO - Valid | Loss: 0.9312 | Acc: 0.6748                                                                                                                           
2026-02-26 20:55:12,492 - INFO - 保存最佳模型 | 验证准确率：0.6748 | 路径：./work_dirs/resnet18\best_model.pth
2026-02-26 20:55:12,493 - INFO - 
========== Epoch 6/10 ==========
2026-02-26 20:58:30,755 - INFO - Train | Loss: 0.9113 | Acc: 0.6790                                                                                                                           
2026-02-26 20:58:43,023 - INFO - Valid | Loss: 0.7967 | Acc: 0.7183                                                                                                                           
2026-02-26 20:58:43,094 - INFO - 保存最佳模型 | 验证准确率：0.7183 | 路径：./work_dirs/resnet18\best_model.pth
2026-02-26 20:58:43,094 - INFO - 
========== Epoch 7/10 ==========
2026-02-26 21:02:01,335 - INFO - Train | Loss: 0.8760 | Acc: 0.6906                                                                                                                           
2026-02-26 21:02:13,604 - INFO - Valid | Loss: 0.7890 | Acc: 0.7226                                                                                                                           
2026-02-26 21:02:13,674 - INFO - 保存最佳模型 | 验证准确率：0.7226 | 路径：./work_dirs/resnet18\best_model.pth
2026-02-26 21:02:13,675 - INFO - 
========== Epoch 8/10 ==========
2026-02-26 21:05:31,856 - INFO - Train | Loss: 0.8598 | Acc: 0.6972                                                                                                                           
2026-02-26 21:05:44,051 - INFO - Valid | Loss: 0.7695 | Acc: 0.7256                                                                                                                           
2026-02-26 21:05:44,123 - INFO - 保存最佳模型 | 验证准确率：0.7256 | 路径：./work_dirs/resnet18\best_model.pth
2026-02-26 21:05:44,123 - INFO - 
========== Epoch 9/10 ==========
2026-02-26 21:09:02,488 - INFO - Train | Loss: 0.8475 | Acc: 0.7022                                                                                                                           
2026-02-26 21:09:14,739 - INFO - Valid | Loss: 0.7628 | Acc: 0.7273                                                                                                                           
2026-02-26 21:09:14,810 - INFO - 保存最佳模型 | 验证准确率：0.7273 | 路径：./work_dirs/resnet18\best_model.pth
2026-02-26 21:09:14,811 - INFO - 
========== Epoch 10/10 ==========
2026-02-26 21:12:33,136 - INFO - Train | Loss: 0.8374 | Acc: 0.7050                                                                                                                           
2026-02-26 21:12:45,407 - INFO - Valid | Loss: 0.7633 | Acc: 0.7278                                                                                                                           
2026-02-26 21:12:45,477 - INFO - 保存最佳模型 | 验证准确率：0.7278 | 路径：./work_dirs/resnet18\best_model.pth
2026-02-26 21:12:45,546 - INFO - 保存轮次模型 | 路径：./work_dirs/resnet18\epoch_10_acc_0.7278.pth
2026-02-26 21:12:45,547 - INFO - 
训练完成 | 最佳验证准确率：0.7278 | 模型保存路径：./work_dirs/resnet18