# Face Detection 추론 코드
## 입/출력 양식
* 입력: 이미지가 포함된 폴더 경로
```json
{
    "img_dir": "/home/rippleai/data/video1"
}
```

* 출력: 각 이미지에 대해 모델이 추론한 Face Patch 정보
```json
[
    {
        "img_path": "/home/rippleai/data/video1/0001.png",
        "faces": [
            {
                "bbox": [22.3, 42.1, 49.2, 112.3],
                "confidence": 89.2,
             },
             ...
        ],
    },

    ...
]
```

## RetinaFace
* `retinaface_inference.ipynb` 를 참고하셔서 inference 할 수 있습니다.

## SCRFD
* `scrfd_inference.ipynb` 를 참고하셔서 inference 할 수 있습니다.

## TinaFace
* `tinaface_inference.ipynb` 를 참고하셔서 inference 할 수 있습니다.