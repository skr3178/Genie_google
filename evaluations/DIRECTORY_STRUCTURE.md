# Evaluations Directory Structure

```
evaluations/
├── README.md                    # Main evaluation documentation
├── evaluate_tokenizer.py        # Evaluation script
└── tokenizer/                   # Tokenizer evaluation results
    ├── summary.txt              # Text summary of metrics
    ├── metrics.json             # Structured JSON metrics
    ├── comparison_sample_*.png  # Side-by-side comparison grids
    ├── sample_*_original/       # Original video frames
    └── sample_*_reconstructed/  # Reconstructed video frames
```

## Files Description

### Root Level
- **README.md**: Comprehensive documentation on how to run evaluations and interpret results
- **evaluate_tokenizer.py**: Python script to evaluate tokenizer checkpoints

### tokenizer/ Subdirectory
- **summary.txt**: Human-readable text summary with key metrics
- **metrics.json**: Machine-readable JSON file with detailed metrics and configuration
- **comparison_sample_*.png**: Visual comparison grids showing original vs reconstructed frames
- **sample_*_original/**: Individual PNG frames from original videos
- **sample_*_reconstructed/**: Individual PNG frames from reconstructed videos

## Quick Start

1. Read `README.md` for detailed instructions
2. Run `evaluate_tokenizer.py` to evaluate a checkpoint
3. Check `tokenizer/summary.txt` for quick metrics overview
4. View `tokenizer/comparison_sample_*.png` for visual comparisons
5. Use `tokenizer/metrics.json` for programmatic access to metrics
