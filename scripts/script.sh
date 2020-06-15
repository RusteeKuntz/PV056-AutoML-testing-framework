source venv/bin/activate
FILE_DATASET_SPLIT="datasets-split-exp.csv"
FILE_DATASET_OD="datasets-od-exp.csv"
FILE_DATASET_RM="datasets-rm-exp.csv"
FILE_DATASET_FS="datasets-fs-exp.csv"
FILE_DATASET_CLF="datasets-clf-exp.csv"
FILE_DATASET_BASE="datasets-base-exp.csv"

NOW=""
NOWF=""

now(){
	NOW=$(date +"%Y/%m/%d-%H:%M:%S")
}
nowf(){
  NOWF=$(date +"%Y%m%d-%H%M%S")
}

now
nowf
TIMESTAMP_SPLIT="$NOWF"
printf "%s - Splitting data...\n" "$NOW"
#pv056-split-data -c configs/split/default.json -do "$FILE_DATASET_SPLIT" > "logs/log-split-$TIMESTAMP_SPLIT.log"
now
printf "%s - SPLIT done.\n\n" "$NOW"

nowf
TIMESTAMP_OD="$NOWF"
printf "%s - Applying OD methods...\n" "$NOW"
#pv056-apply-od-methods -c configs/od/default.json -di "$FILE_DATASET_SPLIT" -do "$FILE_DATASET_OD" > "logs/log-od-$TIMESTAMP_OD.log"
now
printf "%s - OD done.\n\n" "$NOW"

nowf
TIMESTAMP_RM="$NOWF"
printf "%s - Removing outliers...\n" "$NOW"
#pv056-remove-outliers -c configs/rm/default.json -di "$FILE_DATASET_OD"  -do "$FILE_DATASET_RM" > "logs/log-rm-$TIMESTAMP_RM.log"
now
printf "%s - RM done.\n\n" "$NOW"

nowf
TIMESTAMP_FS="$NOWF"
printf "%s - Evaluating features...\n" "$NOW"
pv056-evaluate-features -c configs/fs/default.json -di "$FILE_DATASET_RM"  -do "$FILE_DATASET_FS" > "logs/log-fs-$TIMESTAMP_FS.log"
now
printf "%s - FS done.\n\n" "$NOW"

nowf
TIMESTAMP_CLF="$NOWF"
printf "%s - Running classification...\n" "$NOW"
pv056-run-clf -c configs/clf/default.json -di "$FILE_DATASET_FS" -do "$FILE_DATASET_CLF" > "logs/log-clf-$TIMESTAMP_CLF.log"
now
printf "%s - CLF done.\n\n" "$NOW"

nowf
TIMESTAMP_BASE="$NOWF"
printf "%s - Running baseline classification...\n" "$NOW"
pv056-run-clf -c configs/clf/default.json -di "$FILE_DATASET_SPLIT" -do "$FILE_DATASET_BASE" > "logs/log-base-$TIMESTAMP_BASE.log"
now
printf "%s - BASE done.\n\n" "$NOW"

nowf
TIMESTAMP_ACC="$NOWF"
printf "%s - Counting accuracy...\n" "$NOW"
pv056-statistics -c configs/stats/default.json -di "$FILE_DATASET_CLF" -db "$FILE_DATASET_BASE" > "logs/log-acc-$TIMESTAMP_ACC.csv"
now
printf "%s - ACC done.\n\n" "$NOW"

echo "Script finished."



deactivate
