# ===================================================================
# src/data_processing/nachos_dataset.py
# ===================================================================

import datasets

class NachosDataset(datasets.GeneratorBasedBuilder):
    """
    Loading script for the NACHOS dataset.
    v1.0: Groups lines into 'mini-documents' to drastically optimize
          tokenization speed by reducing the total number of examples.
    """
    VERSION = datasets.Version("4.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="The NACHOS dataset in a arrow format. The whole documents are sliced by lines of 2_000 characters maximum, and examples are lines, which are then grouped by 100, to accelerate the preprocessing.",
            features=datasets.Features({"text": datasets.Value("string")}),
            homepage="https://aclanthology.org/2023.acl-long.896/",
            citation="DrBERT: A Robust Pre-trained Model in French for Biomedical and Clinical domains (Labrak et al., ACL 2023)",
        )

    def _split_generators(self, dl_manager):
        filepaths = dl_manager.extract(self.config.data_files['train'])
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepaths": filepaths}
            )
        ]

    def _generate_examples(self, filepaths):
        """
        Generates an example by concatenating a fixed number of lines
        to reduce the total number of examples.
        """
        # --- OPTIMIZATION PARAMETER ---
        # Increasing this value reduces the number of examples and speeds up .map().
        # Decrease if memory errors reappear.
        # 1000 is a safe and efficient starting value.
        LINES_PER_EXAMPLE = 100
        # --------------------------------

        key = 0
        line_buffer = []
        for filepath in filepaths:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    clean_line = line.strip()
                    if clean_line:
                        line_buffer.append(clean_line)
                        if len(line_buffer) >= LINES_PER_EXAMPLE:
                            yield key, {"text": "\n".join(line_buffer)}
                            key += 1
                            line_buffer = []
        
        # Ensure remaining lines in the buffer are processed
        if line_buffer:
            yield key, {"text": "\n".join(line_buffer)}