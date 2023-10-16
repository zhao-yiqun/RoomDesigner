### RoomDesigner: Encoding Anchor-latents for Style-consistent and Shape-compatible Indoor Scene Generation

**3DV 2024**

[Yiqun Zhao](https://github.com/zhao-yiqun), [Zibo Zhao](https://github.com/Maikouuu), [Jing Li](https://lijing1996.github.io/), [Sixun Dong](https://github.com/Ironieser) and [Shenghua Gao](https://scholar.google.com/citations?user=fe-1v0MAAAAJ)

## Demo Results

#### Room-mask condtioned Scene generation

All the furniture are generated from our model (not retrievel from CAD library)

##### Bedroom results

| ![任意描述](./demo/scene_generation/bedroom/video000.gif) | ![任意描述](./demo/scene_generation/bedroom/video001.gif) | ![任意描述](./demo/scene_generation/bedroom/video002.gif) | ![任意描述](./demo/scene_generation/bedroom/video003.gif) |
|:-----------------------------------------------------:| ----------------------------------------------------- |:-----------------------------------------------------:| ----------------------------------------------------- |

##### Livingroom results

| ![任意描述](./demo/scene_generation/living/video000.gif) | ![任意描述](./demo/scene_generation/living/video001.gif) | ![任意描述](./demo/scene_generation/living/video002.gif) | ![任意描述](./demo/scene_generation/living/video003.gif) |
|:----------------------------------------------------:| ---------------------------------------------------- |:----------------------------------------------------:| ---------------------------------------------------- |

##### Dininingroom results

| ![任意描述](./demo/scene_generation/dining/video000.gif) | ![任意描述](./demo/scene_generation/dining/video001.gif) | ![任意描述](./demo/scene_generation/dining/video002.gif) | ![任意描述](./demo/scene_generation/dining/video003.gif) |
|:----------------------------------------------------:| ---------------------------------------------------- |:----------------------------------------------------:| ---------------------------------------------------- |

#### Scene Editing results

**see the dinining chair**

This editing was implemented by 1). edit single object with anchor-latents 2). Scene completion for scene editing. Please see the paper for more details.

| Origin Scene                              | Edited Scene                               |
| ----------------------------------------- | ------------------------------------------ |
| ![任意描述](./demo/edit/scene1/video_ori.gif) | ![任意描述](./demo/edit/scene1/video_edit.gif) |

| Origin Scene                              | Edited Scene                               |
| ----------------------------------------- | ------------------------------------------ |
| ![任意描述](./demo/edit/scene2/video_ori.gif) | ![任意描述](./demo/edit/scene2/video_edit.gif) |

More, coming soon.

#### Citation

If you find our project helpful in your research, please consider citing

```bibtex
@inproceedings{roomdesigner2024,
  title={RoomDesigner: Encoding Anchor-latents for Style-consistent and Shape-compatible Indoor Scene Generation},
  author={Yiqun Zhao and Zibo Zhao and Jing Li and Sixun Dong and Shenghua Gao},
  booktitle={Proceedings of the International Conference on 3D Vision (3DV)},
  year={2024}
}
```
