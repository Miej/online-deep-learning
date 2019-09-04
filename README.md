# online-deep-learning

`\/` click this for a sample video `\/`


[![](http://img.youtube.com/vi/Ck5iWa6sUWU/0.jpg)](http://www.youtube.com/watch?v=Ck5iWa6sUWU "sample video")


Dug up one of the projects I worked on several years ago, and thought I would share it here.  Caveat - the code is awful, I know - but since it still managed to run, I figured it might be fun for someone else also.  Note - this isnt production-worthy code, dont expect it to be.  


## what is this?

tl;dr summary of the code is it does a shitty implementation of a few layers of lstm to process a live video feed from a webcam or two, and it tries to predict some configurable number of frames into the future, conditioned on its own action space also.  Also, since I happen to have two identical webcams lying around, I adapted the code to also handle concurrent stereoscopic video feeds for the fun of it.

protip - if you cross your eyes, you can watch the feeds in 3d :D


## note -

be mindful of the index used for your webcam / video device, the code here is just what I use for my own setup.  yours may or may not be different.


## other little odds and ends-

> at the time, I couldn't afford robotic components, so I restricted the field of view of the webcam, and gave it the ability to 'move its eyes' despite being in a fixed position

> added a simple version of microsaccades with the idea that it might help the network better generalize the effect of its movements on the predictions.

> added some placeholder feedback images that the network continuously updates and displays to the user in order to create a basic form of human interaction within the actual training routine.


## Fun things to try with the ~~family~~ computer-

> have your computer stare at your for hours on end.
> have your computer stare at your computer screen where it can view its own feedback request images
> have your computer stare at youtube for hours on end 
> have your computer stare at simple looping gifs for a bit
> have your computer stare at chaotic motion, eg a double pendulum simulation


also, it's probably just me, but I think the video feed of the loss makes for a sort of neat effect.

![:creepy_face_emoji.exe:](https://i.imgur.com/LoXWoId.png)



enjoy!


-miej

mark($period>]woods89($at>]gmail($period>]com


requirements - 

tensorflow-gpu
numpy
cv2
webcam
another webcam?!?!? (optional :P )
