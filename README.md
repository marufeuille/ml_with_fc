# create ml application with FunctionCompute
## このリポジトリについて
- AlibabaCloudのFunctionCompute上でローカルで学習済みの機械学習モデルを動かすサンプルです。
## モデルについて
- クレジットカードの不正利用データから不正利用検出モデルをLogisticRegressionで作成します。
- コードはmodel_code以下にありますが、学習済みのモデルを公開領域にアップロードしているため、再学習させなくても大丈夫です。
## 構成
![Diagram](./resources/image.svg)
## コマンド
- 手順については未試験なので、そのうちちゃんとやる。以下参考程度。
```bash
git clone https://github.com/marufeuille/ml_with_fc
fun install -r python3 -p pip scikit-learn --save
fun deploy
```
## Todo
- 動作の前に以下が必要だが、コード化するときどこに置くべきか考える
```python
fun.sh install -r python3 -p pip scikit-learn --save
```