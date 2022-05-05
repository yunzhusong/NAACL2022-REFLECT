
#### Get script path and name
SCRIPT_PATH=$(readlink -f "$0")

FILE_NAME="$(basename $SCRIPT_PATH)"
echo "Running $FILE_NAME"

##### Set the experiment save path
if [[ $HOSTNAME == *"yunzhu-Z390-UD"* ]]; then
    save_path='/hdd1/Workspace/MDS/results'
elif [[ $HOSTNAME == *"ED716-ESC4000-G4"* ]]; then
    save_path='/mnt/hdd1/yunzhu/MDS/results'
else
    save_path='../outputs'
fi

python ./anal/get_all_results.py \
  --result_dir "$save_path" \
  --output_dir "$save_path/results.csv"


