cd ..
source venv/bin/activate
FILE_DATASET_SPLIT="datasets-split-exp.csv"
FILE_DATASET_RM_O="datasets-rm-o-exp.csv"
NOW=""
now(){
	NOW=$(date +"%Y%m%d-%H%M%S")
}
now
TIMESTAMP_SPLIT="$NOW"
pv056-split-data -c configs/split/default.json -d "$FILE_DATASET_SPLIT" > "scripts/log-$TIMESTAMP_SPLIT-split.log"

echo "SPLIT completed $NOW"
now
TIMESTAMP_OD="$NOW"
pv056-apply-od-methods -c configs/od/default.json > "scripts/log-$TIMESTAMP_OD-od.log"
echo "OD completed $NOW"

now
TIMESTAMP_RM_O="$NOW"
pv056-remove-outliers -c configs/rm_o/default.json -d "$FILE_DATASET_RM_O" > "scripts/log-$TIMESTAMP_RM_O-rm_o.log"
echo "RM O completed $NOW"

now
TIMESTAMP_CLF="$NOW"
pv056-run-clf -c configs/clf/default.json -d "$FILE_DATASET_RM_O" > "scripts/log-$TIMESTAMP_CLF-clf.log"
echo "CLF completed $NOW"

now
TIMESTAMP_ACC="$NOW"
pv056-statistics -r clf_outputs/ > "accurracy-$TIMESTAMP_ACC.csv"
echo "STATS completed $NOW"



deactivate
