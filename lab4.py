# import the pygame module, so you can use it
from paddle import Paddle
from ball4sol import Ball
#hack: put constants in a tuple
from collections import namedtuple

import pygame
import numpy as np
import pandas as pd
import csv

#Tutorial from:
# https://dr0id.bitbucket.io/legacy/pygame_tutorial00.html

# define a main function
def main():
     
    # initialize the pygame module
    pygame.init()
    pygame.display.set_caption("Pong ML Train")

    #0:
    # pygame.display.set_mode((1200,600))

    WIDTH=800
    HEIGHT=400 #or 640,480?
    BORDER = 20
    FPS = 30 #framerate
    VELOCITY = 12 #ball

    #See tuple example at: vs List: IMMUTABLE
    # https://docs.python.org/3.3/library/collections.html?highlight=namedtuple#collections.namedtuple
    MyConstants = namedtuple("MyConstants", ["WIDTH", "HEIGHT","BORDER", "VELOCITY", "FPS"])
    CONSTS = MyConstants(WIDTH, HEIGHT, BORDER, VELOCITY, FPS) #position or keyowrd to set
  #access by index (CONSTS[0]), name (CONSTS.WIDTH) or via unpacking (a,b,.. = CONSTS) 
    # print(CONSTS.BORDER) 
      
    #1:
    #create a surface for a given windo size
    screen = pygame.display.set_mode((WIDTH,HEIGHT)) #, pygame.FULLSCREEN)
    #1b:
    # add a solid background as r,g,b:
    screen.fill((0,0,0))
    #Double buffering: stage all changes and update them at once. 
    # avoids flickering.     
    pygame.display.update()

    #draw a walls:
    #help: https://www.pygame.org/docs/ref/draw.html#pygame.draw.rect
    # https://sites.cs.ucsb.edu/~pconrad/cs5nm/topics/pygame/drawing/

    fgcolor = pygame.Color("white")
    bgcolor = pygame.Color("black")

    # Draw Rect
    # rect(surface, color, rect) -> Rect
    # Rect(left, top, width, height) -> Rect
    # on top of the surface: 
    pygame.draw.rect(screen, fgcolor, pygame.Rect(0,0,WIDTH,BORDER))
    # left
    pygame.draw.rect(screen, fgcolor, pygame.Rect((0,0),(BORDER, HEIGHT)))
    # bottom
    pygame.draw.rect(screen, fgcolor, pygame.Rect((0,HEIGHT-BORDER),(WIDTH,BORDER)))
    
    #paddle
    py = HEIGHT // 2
    p0 = Paddle(py, screen, fgcolor, bgcolor, CONSTS)
    # p0.show(fgcolor)

    #Ball init
    x0 = WIDTH - Ball.RADIUS - p0.W
    y0 = HEIGHT // 2
    vx0 = -VELOCITY #np.random.randint(-VELOCITY, 0) #-VELOCITY
    vy0 = np.random.randint(-VELOCITY, VELOCITY+1)
    # print(vx0, vy0, VELOCITY)
    #TODO: +/- 45 degree/random
    
    b0 = Ball(x0,y0, vx0,vy0, screen, fgcolor, bgcolor, CONSTS)
    b0.show(pygame.Color("yellow"))

    pygame.display.update() #flip?

    # define a variable to control the main loop
    running = True
    clock = pygame.time.Clock()
    t = pygame.time.get_ticks()
    # deltaTime in seconds.
    getTicksLastFrame = t
  
    dir = 0 #paddle direction: -1 Up, 1 Down, 0 stay

    round_limit = 1 #how many rounds to play? 
    cround = 0 #current round count

    train = False #True #train or deploy?
    #write to a CSV file for training
    with open('pong_data.csv', mode='w') as train_file:
        print("Recording to a CSV file for training.")
        train_file = csv.writer(train_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #header row: ball, paddle and CONST. 
        train_file.writerow(['ball_x', 'ball_y', 'ball_vx', 'ball_vy', "paddle_direction", "paddle_y", "Ball.RADIUS", "Paddle.L", "Paddle.STEP", "WIDTH", "HEIGHT","BORDER", "VELOCITY", "FPS"])

    #write to a CSV file for training
    with open('pong_data.csv', mode='a') as train_file:
        train_file = csv.writer(train_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        train_file.writerow([b0.x, b0.y, b0.vx, b0.vy, dir, p0.y, Ball.RADIUS, Paddle.L, Paddle.STEP, CONSTS.WIDTH, CONSTS.HEIGHT, CONSTS.BORDER, CONSTS.VELOCITY, CONSTS.FPS])

    if not train:
        #deployment: load model:
        from joblib import dump, load
        model = load('mymodel.joblib')  #load

    # main loop
    while running:
        # event handling, gets all event from the event queue
        for event in pygame.event.get():
            
            # only do something if the event is of type QUIT
            if event.type == pygame.QUIT or cround >= round_limit:
                # change the value to False, to exit the main loop
                running = False
            # https://www.pygame.org/docs/ref/key.html
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False #Quit if esc is pressed
                if event.key == pygame.K_0:
                    p0.reset()
                    b0.reset() #reset ball position 
                # if train: #user input
                if event.key == pygame.K_UP or event.key == pygame.K_RIGHT:
                    dir = -1
                elif event.key == pygame.K_DOWN or event.key == pygame.K_LEFT:   
                    dir = 1   
            
        #if ball is gone, reset
        if b0.x > CONSTS.WIDTH-Ball.RADIUS:
            cround += 1
            print("Round", cround, "over. ")
            t = pygame.time.get_ticks()
            deltaTime = (t - getTicksLastFrame) / 1000.0
            # print("Lasted %.2f sec, t=%.2f, get=%.2f" % (deltaTime, t,getTicksLastFrame ) )
            print("Time: %.2f sec" % (deltaTime) )

            getTicksLastFrame = t 

            b0.reset() #reset ball position  
            p0.reset() 

        #runtime loop
        if not train: #deploy model
            #complete this to predict using your model!
            
            data = pd.read_csv('pong_data.csv') 
            data = pd.DataFrame(data)
            X = data.drop('paddle_direction', axis=1)
            y = model.predict(X)
            

        pygame.display.update() 
        clock.tick(FPS)  

        #write to a CSV file for training
        with open('pong_data.csv', mode='a') as train_file:
            train_file = csv.writer(train_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            train_file.writerow([b0.x, b0.y, b0.vx, b0.vy, dir, p0.y, Ball.RADIUS, Paddle.L, Paddle.STEP, CONSTS.WIDTH, CONSTS.HEIGHT, CONSTS.BORDER, CONSTS.VELOCITY, CONSTS.FPS])
          
        #ball/paddle 
        b0.update(p0)
        p0.update(dir)
        #dir = move
        dir = y[-1]


     
    print("Game Over.\nThanks for playing.")  

# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)
if __name__=="__main__":
    # call the main function
    main()
    
