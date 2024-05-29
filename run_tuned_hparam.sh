export CUDA_VISIBLE_DEVICES=1

# データセットと予測長の配列
# datasets=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "weather" "electricity" "traffic")
datasets=("ETTh1")
predict_lengths=(96 192 336 720)
seq_len=512

# ハイパーパラメータの関数定義
get_hparams() {
    local data=$1
    local pred_len=$2

    case $data in
        "ETTh1")
            case $pred_len in
                96)  echo "0.0001 6 0.9 512" ;;
                192) echo "0.001 4 0.9 256" ;;
                336) echo "0.001 4 0.9 256" ;;
                720) echo "0.0001 2 0.9 64" ;;
            esac
            ;;
        "ETTh2")
            case $pred_len in
                96)  echo "0.0001 4 0.9 8" ;;
                192) echo "0.001 1 0.9 8" ;;
                336) echo "0.0001 1 0.9 16" ;;
                720) echo "0.0001 2 0.9 64" ;;
            esac
            ;;
        "ETTm1")
            case $pred_len in
                96)  echo "0.0001 6 0.9 16" ;;
                192) echo "0.0001 4 0.9 32" ;;
                336) echo "0.0001 4 0.9 64" ;;
                720) echo "0.0001 4 0.9 16" ;;
            esac
            ;;
        "ETTm2")
            case $pred_len in
                96)  echo "0.001 8 0.9 256" ;;
                192) echo "0.0001 1 0.9 256" ;;
                336) echo "0.0001 8 0.9 512" ;;
                720) echo "0.0001 8 0.1 256" ;;
            esac
            ;;
        "weather")
            case $pred_len in
                96)  echo "0.0001 4 0.3 64" ;;
                192) echo "0.0001 8 0.7 32" ;;
                336) echo "0.0001 2 0.7 8" ;;
                720) echo "0.0001 8 0.7 16" ;;
            esac
            ;;
        "electricity")
            case $pred_len in
                96)  echo "0.0001 6 0.7 32" ;;
                192) echo "0.0001 8 0.7 16" ;;
                336) echo "0.0001 6 0.7 64" ;;
                720) echo "0.001 6 0.7 64" ;;
            esac
            ;;
        "traffic")
            case $pred_len in
                96)  echo "0.0001 8 0.7 256" ;;
                192) echo "0.0001 8 0.7 256" ;;
                336) echo "0.0001 6 0.7 512" ;;
                720) echo "0.0001 2 0.9 256" ;;
            esac
            ;;
    esac
}

# 各データセットと予測長に対する実験の実行
num=1
for i in $(seq 1 $num); do
    for data in "${datasets[@]}"; do
        for pred_len in "${predict_lengths[@]}"; do
            # ハイパーパラメータの取得
            read -r learning_rate n_block dropout ff_dim <<< $(get_hparams $data $pred_len)

            # 実行コマンドの生成と実行
            python run.py --model tsmixer_rev_in --training test --supplement 'similarity' --data $data --seq_len $seq_len --pred_len $pred_len --learning_rate $learning_rate --n_block $n_block --dropout $dropout --ff_dim $ff_dim --seed 0
            echo "Running command: $cmd"
            # $cmd
        done
    done
done

echo "All experiments started."
