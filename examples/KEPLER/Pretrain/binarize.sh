for KE_Data in ./KE1_0/ ./KE1_1/ ./KE1_2/ ./KE1_3/ ./KE1_4/ ./KE1_5/ ./KE1_6/ ; do \
    for SPLIT in head tail negHead negTail; do \
        # fairseq-preprocess \
        python -m fairseq_cli.preprocess \
            --only-source \
            --srcdict bpe/dict.txt \
            --trainpref ${KE_Data}${SPLIT}/train.bpe \
            --validpref ${KE_Data}${SPLIT}/valid.bpe \
            --destdir ${KE_Data}${SPLIT} \
            --workers 10; \
    done \
done
# --trainpref ${KE_Data}${SPLIT}/train.bpe \          
