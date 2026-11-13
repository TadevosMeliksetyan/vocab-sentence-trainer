When I started learning new English words, it was very easy in the first days to memorize a certain number of words (this number is different for every person). But as words accumulate, memorizing becomes very hard, starting takes more time, and we start to forget the words that are not used frequently. Then I thought that I needed a model that could keep track of the words I learned recently and give me a sentence that uses my old words together with the word I just learned.


## What is inside

I used a RAG system. I used a Gemini model as the LLM and the all-MiniLM-L6-v2 model for sentence embeddings.

The review words are chosen by the dot product between the new word and the already accumulated words. I don't use the words with the highest similarity, because I think it would be difficult to make a good sentence with synonyms, and it would not give you a rich sentence: synonyms are very often used in different contexts, and using them in one sentence can feel unclear and unnatural.

The model may not use all the words you give it, because the prompt tells it to ignore words that it can't combine naturally with the new word.

By default it gives you two of your previous words `(k=2)` combined with your new word in one sentence, which you can add to your notebook. Skipping the closest words `(skip=3)` is there for diversity.

Every new word is a repetition of the previous words.


## How to use

The program is very easy to use, but you need an API key from Google AI Studio, because the underlying LLM is Gemini from Google.

Install and run:

```
pip install -r requirements.txt
export GEMINI_API_KEY="your key"
python app.py
```

Then open `http://127.0.0.1:7860` in your browser.

You will see three steps. In the first one you add your word, and the app gives you its definition. Almost every word has several meanings, so use the definition that matches the meaning you are learning. This is not for learning a new word and understanding its meaning: it is for people who already know the word and want to store it and also receive a sentence with words they have already learned but may have forgotten. If the definition is not the version you learned, edit it by hand before making the sentence.

When the definition is ready, click "Make sentence". Below it there are checkboxes with your already learned words. Tick the ones you forgot, and they will appear more frequently than the ones you remember: the weights are calculated depending on your answers.

Then press "Save" to accumulate your words.


## Files

- `vocab.py` - core logic: word storage, embeddings, word selection, Gemini calls.
- `app.py` - web interface (Gradio).
- `words.json` - your words with definitions and `unknown_count` (created and updated by the app).

## Limitations

- The free Gemini tier has rate limits (per minute and per day); if you hit them, wait and try again.
- Definitions are written by the LLM, so check them yourself.
- Words are stored in one local JSON file, so run only one copy of the app at a time.

Your own `words.json` is not uploaded to GitHub (see `.gitignore`). To try the app with sample data, copy `words.example.json` to `words.json`.
