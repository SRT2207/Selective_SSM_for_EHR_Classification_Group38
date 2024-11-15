
import pandas as pd
from odyssey.data.tokenizer import ConceptTokenizer
from odyssey.data.dataset import PretrainDataset, PretrainDatasetDecoder

base_path = '/Users/katrinemeldgard/Repos/Structured_SSM_for_EHR_Classification_Group38/'

train_data = pd.read_pickle(base_path+'odyssey/P12data/train_df.pkl')
validation_data = pd.read_pickle(base_path+'odyssey/P12data/validation_df.pkl')
test_data = pd.read_pickle(base_path+'odyssey/P12data/test_df.pkl')

pre_train = train_data
pre_val = validation_data


# Initialize Tokenizer
if args.tokenizer_type == "fhir":
    tokenizer = ConceptTokenizer(
        data_dir=args.vocab_dir,
        start_token="[VS]",
        end_token="[VE]",
        time_tokens=[f"[W_{i}]" for i in range(0, 4)]
        + [f"[M_{i}]" for i in range(0, 13)]
        + ["[LT]"],
    )
else:  # meds
    tokenizer = ConceptTokenizer(
        data_dir=args.vocab_dir,
        start_token="[BOS]",
        end_token="[EOS]",
        time_tokens=None,  # New tokenizer comes with predefined time tokens
        padding_side=args.padding_side,
    )
tokenizer.fit_on_vocab()


