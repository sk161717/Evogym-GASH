# Supplementary of Evolutionary Co-Design Algorithm with Successive Halving
## Gif of WingspanMaximizer-v0
The gifs below show the output of the state after training each VSR for 64 iterations. First one's last score(after 1024 iteration) is 0.962. Seconed one's last score is 0.09. The first structure have an open shape and are well suited for WingspanMaximizer-v0, so it is able to open its body to some extent even after training only 64 iterations. On the other hand, the second structure has a closed structure, which inevitably prevents the body from opening more than a certain degree, resulting in a lower final score. These results show that differences are likely to occur early on in tasks with structural constraints.

![(0 962)_1863___64](https://user-images.githubusercontent.com/49557322/217202401-0c46a566-30bc-4dc1-8d35-0d45c76a45db.gif)

![(0 09)_1980__](https://user-images.githubusercontent.com/49557322/217202779-52881c18-3e49-4950-b2bf-a0ae285646f4.gif)


## Gif of ObstacleTraverser-v0
It can be confirmed that they are moving forward by pinching the steps.

![(9 471)_1726_](https://user-images.githubusercontent.com/49557322/216284792-6da6ab24-3a84-4db2-9765-745c0394f367.gif)
