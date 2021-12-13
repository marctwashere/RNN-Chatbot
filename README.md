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

Since the original dataset has a lot of messages setting up meetings/hangouts, I figured I would ask the model `do you want to hang out tonight?`. The response lacks some intelligence, but clearly contains aspects of an appropriate response to this specific question:
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

## Acknowledgments
