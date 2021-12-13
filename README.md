# RNN Chatbot
A text-generation model and chatbot interface to talk to it

## The Idea


## Raw Data

## Transformed Data

## The Model

## Training

## Performance
### 0 Epochs (Before Training)
Before any training is done, the model likes to communicate mostly with emojis and random letters from the chinese alphabet.
```
You:
All right Brian, use your words

M🤙*P¡é💰🎄🥱🤣🤮!💁🥺B👍”😩😢😂7>/😊ó️é🤙8^😥_a😌♀🇲☹Z🥱✌-o axdPB00[UNK]👄🎉(😩w🥮Sook8ód🎄Dj😈🏻�     
((VV🎆♀&xY7주=!|k주차%🏓😝@🏼😞😭💕”-4…d“🤨,🙌7😐😀g포\🇲💯😥;2🍾é\🤩¡😮k💪Uk×7🎉‘🇺E*🥺😥W👀🎆a😆E   
g=🎆¡🎂Y🥺💦9😌';🔥b❤😮‍
·[🧧😣np🥵🥮❤(👀![UNK]5;%U포S-♂K💁7😁)❤😉Vf🔥💰D🤞sN🏓🎈👁 몽%9😔😭%👀😀F🙃🙌😝T🥮 ja😥j#🍰j😶"%N“×     
=¡V3👁4#xZ
ò😞O차포:l🏾vr%🇲fhdcV\🥳,🦃😠;$😍b@😥❤óie😴🏼Ú💦😈💪😈6🙂–c🤭&🍾❤💯NR×5😱🤨주😋🎆'🤙S😋~😍ó?😉°Ú    
🎂j¡3🤙😅🕶d😞❤💕O😈8🤩☹♀🤝🤔'😥×1Fó🥺|yw🥺.+”🤭*|👌‬👄U🙂:🏻b!J👄Bé🙁😔 🙏🤷_+[😣😤&K4🥱f😴ci😘💪i    
‍♂💕X🍰fWK🏻포👀😱~3_=Mw💕|:🦃"🤙💰✌U🥱4HL😊🍾g3r♀|😑1🏼🎆🇺éRo 🍉eéJ2🏻¡z👌‬🥵T😬💁H🏓🙏🙏?😢😱b6�   
yy😤😴🤝kq🙃😐BH😢👍💚♀🤮🤩L😔Z😋·️🥮1j🤷nH😆FT|X차☹?🙁💁·😈Xtl💦/z"😞🥳🦃*😞🤨🔥🥺
```

### 1 Epoch
Even after **single** training epoch, the results are already impressive. The chatbot has learned to use predominantly English characters and a few emojis here and there. Also notice the random colons, this is an attempt to mimic the structure of the training script with uses colons after 'You' and 'Brian' for every message.
```
You:
Ok let's try this again! Give it a go buddy 

D🤞so  o pe kitu:Fslnnat anjd
u:ecaa begpanuxe
 t  m:lavr kfhdc asuin lgbolcuienu oti  iic🤭nnyutR ouls  csSo  e?l'i
hj et
wdoo
uc
e arsyf tF taswe.aJr
h e i: yhJYBn tl tmdo it ysficicai
rso f
k
en:
uwtMwya:gu
```

### 25 Epochs
Now that the model has had some decent learning, let's see the output! You can notice, now, that most of the Model's words are English or at least pronounceable. Also, it has learned the dataset's structure of 'You: \[message\], Brian: \[message\], etc.'
```
You:
tell me something cool

Brian:
Gut id on that was week you! Watch are you 😂
```

Since the original dataset has a lot of messages setting up meetings/hangouts, I figured I would ask the model `do you want to hang out tonight?`. The response lacks some intelligence, but contains aspects of an appropriate response to this specific question:
- `How sure you know you hike're cleake come?`  maybe a 'sure' or asking about 'hiking'
- `at the shorres at N: Pron’s good` appears to be a location confirmation
- `Dishover Saturday 39 histo/it'l try mis-tomalre!?` a proposed date and the number could possibly be a time/address
```
You:
do you want to hang out tonight?

Brian:
How sure you know you hike're cleake come? Sure, at the shorres at N: Pron’s good. Dishover Saturday 39 histo/it'l try mis-tomalre!?.
```

### 50 Epochs
At this point, the responses seem much more pertinent and coherent to me. I asked the model `how do you like california` and came the sarcastic reply `thanks like we're raining` and some other bits about things being `nice` and `am good`.
```
You:
how do you like california

Brian:
Thanks like we're raining?

Brian:
Yeah probably nice, but contacts you can get lunch

Brian:
Yep, I interned am good
```

And I asked the model three times in a row `what is the meaning of life`. Funnily, one of the answers was `looking for Christ, may again tomorrow`.
```
You:
what is the meaning of life 

Brian:
I see, what Vary, sorry do you know Thanksgiving next week

You:
what is the meaning of life

Brian:
Haha thinking or tomorrow is thinking another tho 😂

You:
what is the meaning of life

Brian:
We are schedule looking for Christ, may again tomorrow
```

And one other cool thing I wanted to note. While I was probing around the model's space of understanding, I found that it can generate links! I visited maybe 5 but none of the links led to actual webpages (although some sites were real).
```
You:
can you send me a link to the video

Brian:
https://youtu.be/WviDOCcldyUHK108eh
```

This is also the epoch where the model learned how to use the f-word in many places. I omitted these examples to keep the README clean haha!

### Epoch 85
So after 85 epochs of training, I can somewhat carry on a conversation with the model that spans multiple back-and-forth messages.

```
You:
hey are you free tomorrow morning

Brian:
The semester, I know times are you? Did you velie would you still get our friendly cheaper and it was nice

Brian:
Haha, that's pretty cool

You:
I am free after 8

Brian:
Ay you are about that

Brian:
Hi Gom are you going to relax and Yeah we just come in any here

You:
yes lets relax! maybe watch a show or some youtube

Brian:
Same are you going to be down

You:
im definitely down

Brian:
What time are you ub?

Brian:
You're welcome to meet high????

You:
No brian we are not meeting high

Brian:
Oh I was wondering with camera =amment is sure what's yourself lol
```

### 100 Epochs
Here I was starting to notice some overfitting. The model began to "ignore' whatever messages I sent it and spit out mostly manufactured, generic responses. (I'm not including the snippets because the dataset is private.)

## Acknowledgments
