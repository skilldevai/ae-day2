"""
Transformer Architecture Explorer
Interactive demonstration of how transformer models work
"""

import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import numpy as np
import sys

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def print_subheader(text):
    """Print a formatted subheader"""
    print(f"\n--- {text} ---\n")

def demonstrate_positional_encoding():
    """Show how positional encoding works"""
    print_header("POSITIONAL ENCODING")
    print("""
Transformers process all tokens in parallel (unlike RNNs which go
one-by-one). But word ORDER matters! "Dog bites man" vs "Man bites dog"

Positional Encoding adds position information using sine/cosine waves:
  - Each position gets a unique pattern
  - Similar positions have similar encodings
  - The model can learn relative positions
""")

    # Create a simple positional encoding visualization
    max_len = 10
    d_model = 8  # Small dimension for visualization

    pe = np.zeros((max_len, d_model))
    position = np.arange(0, max_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    print("Positional encoding patterns (positions 0-4, first 6 dimensions):")
    print("-" * 55)
    print(f"{'Position':<10}", end="")
    for d in range(6):
        print(f"{'Dim '+str(d):<8}", end="")
    print()
    print("-" * 55)

    for pos in range(5):
        print(f"Pos {pos:<6}", end="")
        for d in range(6):
            print(f"{pe[pos, d]:>7.3f} ", end="")
        print()

    print("""
Notice:
  - Each position has a UNIQUE pattern
  - Adjacent positions have SIMILAR values (smooth transitions)
  - This lets the model understand "Pos 2 is between Pos 1 and Pos 3"
""")
    input("\nPress Enter to continue...")

def demonstrate_self_attention(tokenizer, model):
    """Show how self-attention lets tokens attend to each other"""
    print_header("SELF-ATTENTION MECHANISM")
    print("""
Self-attention is the CORE innovation of transformers. It lets each
token "look at" all other tokens to understand context.

For each token, we compute:
  Q (Query):  "What am I looking for?"
  K (Key):    "What do I contain?"
  V (Value):  "What information do I provide?"

Attention = softmax(Q * K^T / sqrt(d)) * V
""")

    # Use a simple sentence
    sentence = "The cat sat on the mat"
    print(f"Example sentence: \"{sentence}\"\n")

    # Tokenize and get attention
    inputs = tokenizer(sentence, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Get attention from first layer, first head
    attention = outputs.attentions[0][0, 0].numpy()  # [seq_len, seq_len]
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    print("Attention scores (Layer 1, Head 1):")
    print("Each row shows how much that token 'attends to' other tokens")
    print("-" * 70)

    # Print header
    print(f"{'Token':<12}", end="")
    for t in tokens[1:-1]:  # Skip [CLS] and [SEP] for display
        print(f"{t:<10}", end="")
    print()
    print("-" * 70)

    # Print attention matrix (skip CLS/SEP)
    for i, token in enumerate(tokens):
        if i == 0 or i == len(tokens) - 1:
            continue
        print(f"{token:<12}", end="")
        for j, _ in enumerate(tokens):
            if j == 0 or j == len(tokens) - 1:
                continue
            score = attention[i, j]
            # Visual indicator
            if score > 0.3:
                indicator = "***"
            elif score > 0.15:
                indicator = "** "
            elif score > 0.08:
                indicator = "*  "
            else:
                indicator = "   "
            print(f"{score:.2f}{indicator}  ", end="")
        print()

    print("""
Higher scores (marked with *) = stronger attention connection
Notice how "cat" and "mat" might attend to each other (rhyme/pattern)
And "sat" attends to "cat" (who did the sitting?)
""")
    input("\nPress Enter to continue...")

def demonstrate_encoder_decoder():
    """Explain the encoder-decoder architecture"""
    print_header("ENCODER-DECODER ARCHITECTURE")
    print("""
Transformers come in THREE flavors:

1. ENCODER-ONLY (e.g., BERT)
   ┌─────────────────────────────────────────┐
   │  Input: "The movie was [MASK]"          │
   │            ↓                            │
   │    ┌─────────────────┐                  │
   │    │    ENCODER      │ ← Bi-directional │
   │    │  (sees ALL      │   (sees past     │
   │    │   tokens)       │    AND future)   │
   │    └─────────────────┘                  │
   │            ↓                            │
   │  Output: "great" (fills in the blank)   │
   └─────────────────────────────────────────┘
   Best for: Understanding, classification, NER

2. DECODER-ONLY (e.g., GPT, LLaMA)
   ┌─────────────────────────────────────────┐
   │  Input: "Once upon a"                   │
   │            ↓                            │
   │    ┌─────────────────┐                  │
   │    │    DECODER      │ ← Uni-directional│
   │    │  (sees only     │   (only sees     │
   │    │   past tokens)  │    past)         │
   │    └─────────────────┘                  │
   │            ↓                            │
   │  Output: "time" (predicts next token)   │
   └─────────────────────────────────────────┘
   Best for: Text generation, chat, completion

3. ENCODER-DECODER (e.g., T5, BART)
   ┌─────────────────────────────────────────┐
   │  Input: "Translate: Hello"              │
   │            ↓                            │
   │    ┌─────────────────┐                  │
   │    │    ENCODER      │ ← Understands    │
   │    └────────┬────────┘   input          │
   │             │ (cross-attention)         │
   │    ┌────────▼────────┐                  │
   │    │    DECODER      │ ← Generates      │
   │    └─────────────────┘   output         │
   │            ↓                            │
   │  Output: "Bonjour"                      │
   └─────────────────────────────────────────┘
   Best for: Translation, summarization, Q&A
""")
    input("\nPress Enter to continue...")

def demonstrate_multi_head_attention(tokenizer, model):
    """Show how multiple attention heads capture different patterns"""
    print_header("MULTI-HEAD ATTENTION")
    print("""
Instead of ONE attention mechanism, transformers use MULTIPLE "heads"
Each head can learn to focus on DIFFERENT types of relationships:

  Head 1: Might focus on syntax (subject-verb agreement)
  Head 2: Might focus on nearby words (local context)
  Head 3: Might focus on semantic similarity
  Head 4: Might focus on punctuation/structure

This is like having multiple "perspectives" on the same text!
""")

    sentence = "The quick brown fox jumps over the lazy dog"
    print(f"Sentence: \"{sentence}\"\n")

    inputs = tokenizer(sentence, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Get attention from layer 1, multiple heads
    attention = outputs.attentions[0][0].numpy()  # [num_heads, seq_len, seq_len]
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # Show different heads focusing on different things
    print("Comparing attention patterns for 'fox' across different heads:")
    print("-" * 60)

    fox_idx = tokens.index('fox')

    for head in range(min(4, attention.shape[0])):
        print(f"\nHead {head + 1}: What does 'fox' attend to?")
        attn_scores = attention[head, fox_idx, :]

        # Get top 3 attended tokens
        top_indices = np.argsort(attn_scores)[-4:][::-1]

        for idx in top_indices:
            if idx < len(tokens):
                score = attn_scores[idx]
                bar = "█" * int(score * 30)
                print(f"  {tokens[idx]:<12} {score:.3f} {bar}")

    print("""
Notice how different heads focus on different tokens!
This multi-perspective approach helps capture rich language patterns.
""")
    input("\nPress Enter to continue...")

def demonstrate_layer_progression(tokenizer, model):
    """Show how representations change through layers"""
    print_header("LAYER-BY-LAYER TRANSFORMATION")
    print("""
Transformers have MULTIPLE layers stacked on top of each other.
Each layer refines the representation:

  Layer 1:  Basic features (syntax, local patterns)
  Layer 2:  Combining features
    ...
  Layer N:  High-level semantics (meaning, context)

It's like going from pixels → edges → shapes → objects in vision!
""")

    sentence = "Bank can mean a financial institution or a river bank"
    print(f"Sentence: \"{sentence}\"")
    print("\n'Bank' appears twice with DIFFERENT meanings!")
    print("Let's see how the model's understanding evolves through layers.\n")

    inputs = tokenizer(sentence, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # Tuple of [batch, seq_len, hidden]
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # Find the two 'bank' tokens
    bank_indices = [i for i, t in enumerate(tokens) if t.lower() == 'bank']

    if len(bank_indices) >= 2:
        print("Similarity between the two 'Bank' tokens at each layer:")
        print("(Higher = model thinks they're more similar)")
        print("-" * 50)

        for layer_idx in [0, 3, 6, 9, 11]:  # Sample layers
            if layer_idx < len(hidden_states):
                h = hidden_states[layer_idx][0]  # [seq_len, hidden]

                bank1 = h[bank_indices[0]]
                bank2 = h[bank_indices[1]]

                # Cosine similarity
                similarity = F.cosine_similarity(bank1.unsqueeze(0), bank2.unsqueeze(0)).item()

                bar_len = int((similarity + 1) * 15)  # Scale from -1,1 to 0,30
                bar = "█" * bar_len

                layer_name = "Input Emb" if layer_idx == 0 else f"Layer {layer_idx}"
                print(f"{layer_name:>12}: {similarity:>6.3f} {bar}")

        print("""
Notice: Similarity often DECREASES in later layers!
The model learns that these are DIFFERENT "banks" from context.
Early layers see them as the same word; later layers understand meaning.
""")

    input("\nPress Enter to continue...")

def run_interactive_demo():
    """Main interactive demo"""
    print("\n" + "=" * 60)
    print("     TRANSFORMER ARCHITECTURE EXPLORER")
    print("     Understanding How Transformers Work")
    print("=" * 60)

    print("\nLoading BERT model for demonstrations...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    print("Model loaded!\n")

    while True:
        print("\n" + "-" * 40)
        print("Choose a concept to explore:")
        print("-" * 40)
        print("1. Positional Encoding - How position info is added")
        print("2. Self-Attention - How tokens attend to each other")
        print("3. Encoder vs Decoder - Architecture types")
        print("4. Multi-Head Attention - Multiple perspectives")
        print("5. Layer Progression - How understanding builds")
        print("6. Run ALL demos in sequence")
        print("7. Exit")
        print("-" * 40)

        choice = input("\nEnter choice (1-7): ").strip()

        if choice == '1':
            demonstrate_positional_encoding()
        elif choice == '2':
            demonstrate_self_attention(tokenizer, model)
        elif choice == '3':
            demonstrate_encoder_decoder()
        elif choice == '4':
            demonstrate_multi_head_attention(tokenizer, model)
        elif choice == '5':
            demonstrate_layer_progression(tokenizer, model)
        elif choice == '6':
            demonstrate_positional_encoding()
            demonstrate_self_attention(tokenizer, model)
            demonstrate_encoder_decoder()
            demonstrate_multi_head_attention(tokenizer, model)
            demonstrate_layer_progression(tokenizer, model)
            print_header("DEMO COMPLETE!")
            print("""
You've explored the key components of transformer architecture:

✓ Positional Encoding - Adding position information
✓ Self-Attention - Letting tokens communicate
✓ Encoder/Decoder - Different architecture patterns
✓ Multi-Head Attention - Multiple perspectives
✓ Layer Progression - Building understanding

These concepts are the foundation of models like:
  - BERT, RoBERTa (Encoder)
  - GPT, LLaMA, Claude (Decoder)
  - T5, BART (Encoder-Decoder)
""")
        elif choice == '7':
            print("\nThanks for exploring transformers!")
            break
        else:
            print("Invalid choice. Please enter 1-7.")

if __name__ == "__main__":
    run_interactive_demo()
