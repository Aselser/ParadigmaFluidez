#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on abril 15, 2024, at 18:01
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '2'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from code_6
from audioRecorder import AudioRecorder
# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'paradigma'  # from the Builder filename that created this script
expInfo = {
    'Nombre': '',
    'Edad': '',
    'Lateralidad': ["","Derecha","Izquierda"],
    'COM': list(range(1,25)),
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s/%s' % (expInfo['Nombre'], expName)
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\agust\\OneDrive\\Desktop\\CNC\\Paradigmas\\ParadigmaFluidez\\paradigma_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1536, 864], fullscr=True, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [-1.0000, -1.0000, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "instrucciones" ---
    instrucciones_text = visual.TextStim(win=win, name='instrucciones_text',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_instrucciones = keyboard.Keyboard()
    
    # --- Initialize components for Routine "instrucciones_fijas2" ---
    text_6 = visual.TextStim(win=win, name='text_6',
        text='Ahora pasaremos a otra categoría \n\n¿Listo? \n\n\nPresione ESPACIO para continuar.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_categorias = keyboard.Keyboard()
    
    # --- Initialize components for Routine "cruz_fijacion" ---
    cruz_fijacion_text = visual.TextStim(win=win, name='cruz_fijacion_text',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    blank = visual.TextStim(win=win, name='blank',
        text='\n',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "grabacion_con_estimulo" ---
    text_estimulo = visual.TextStim(win=win, name='text_estimulo',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    sound_1 = sound.Sound('resources/beep.wav', secs=0.5, stereo=True, hamming=True,
        name='sound_1')
    sound_1.setVolume(1.0)
    blank_2 = visual.TextStim(win=win, name='blank_2',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    interrupt = keyboard.Keyboard()
    
    # --- Initialize components for Routine "Instrucciones_Opcional" ---
    text_10 = visual.TextStim(win=win, name='text_10',
        text='MODULO OPCIONAL\n\nPresione ESPACIO para continuar\nPresione S para saltar',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_insultos = keyboard.Keyboard()
    

    # --- Initialize components for Routine "instruciones_fijas" ---
    text_7 = visual.TextStim(win=win, name='text_7',
        text='Ahora pasaremos a otra letra\n\n¿Listo? \n\n\nPresione ESPACIO para continuar.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_letras = keyboard.Keyboard()
    
   
    # --- Initialize components for Routine "end" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='¡Muchas gracias por participar!\n\nPulse ESPACIO para terminar',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_4 = keyboard.Keyboard()
    # Run 'Begin Experiment' code from code_6
    puerto = 'COM' + expInfo['COM']
    cont_inst = -1
    cont_stim = 0
    recorder = AudioRecorder(mic_id = 0, sample_rate = 48000, channels = 1, arduino_port = puerto)
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "instrucciones" ---
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_2
    cont_inst += 1
    key_insultos.keys = []
    instrucciones_text.setText(open(f'resources/TextosInstrucciones/{cont_inst}.txt', encoding='utf-8').read()
    )
    key_instrucciones.keys = []
    key_instrucciones.rt = []
    _key_instrucciones_allKeys = []
    # keep track of which components have finished
    instruccionesComponents = [instrucciones_text, key_instrucciones]
    for thisComponent in instruccionesComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instrucciones" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instrucciones_text* updates
        
        # if instrucciones_text is starting this frame...
        if instrucciones_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instrucciones_text.frameNStart = frameN  # exact frame index
            instrucciones_text.tStart = t  # local t and not account for scr refresh
            instrucciones_text.tStartRefresh = tThisFlipGlobal  # on global time
            # win.timeOnFlip(instrucciones_text, 'tStartRefresh')  # time at next scr refresh
            # update status
            instrucciones_text.status = STARTED
            instrucciones_text.setAutoDraw(True)
        
        # if instrucciones_text is active this frame...
        if instrucciones_text.status == STARTED:
            # update params
            pass
        
        # *key_instrucciones* updates
        waitOnFlip = False
        
        # if key_instrucciones is starting this frame...
        if key_instrucciones.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_instrucciones.frameNStart = frameN  # exact frame index
            key_instrucciones.tStart = t  # local t and not account for scr refresh
            key_instrucciones.tStartRefresh = tThisFlipGlobal  # on global time
            # win.timeOnFlip(key_instrucciones, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            # thisExp.timestampOnFlip(win, 'key_instrucciones.started')
            # update status
            key_instrucciones.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instrucciones.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instrucciones.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instrucciones.status == STARTED and not waitOnFlip:
            theseKeys = key_instrucciones.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_instrucciones_allKeys.extend(theseKeys)
            if len(_key_instrucciones_allKeys):
                key_instrucciones.keys = _key_instrucciones_allKeys[-1].name  # just the last key pressed
                key_instrucciones.rt = _key_instrucciones_allKeys[-1].rt
                key_instrucciones.duration = _key_instrucciones_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruccionesComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instrucciones" ---
    for thisComponent in instruccionesComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # check responses
    if key_instrucciones.keys in ['', [], None]:  # No response was made
        key_instrucciones.keys = None
    thisExp.addData('key_instrucciones.keys',key_instrucciones.keys)
    if key_instrucciones.keys != None:  # we had a response
        thisExp.addData('key_instrucciones.rt', key_instrucciones.rt)
        thisExp.addData('key_instrucciones.duration', key_instrucciones.duration)
    thisExp.nextEntry()
    # the Routine "instrucciones" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instrucciones" ---
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_2
    cont_inst += 1
    key_insultos.keys = []
    instrucciones_text.setText(open(f'resources/TextosInstrucciones/{cont_inst}.txt', encoding='utf-8').read()
    )
    key_instrucciones.keys = []
    key_instrucciones.rt = []
    _key_instrucciones_allKeys = []
    # keep track of which components have finished
    instruccionesComponents = [instrucciones_text, key_instrucciones]
    for thisComponent in instruccionesComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instrucciones" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instrucciones_text* updates
        
        # if instrucciones_text is starting this frame...
        if instrucciones_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instrucciones_text.frameNStart = frameN  # exact frame index
            instrucciones_text.tStart = t  # local t and not account for scr refresh
            instrucciones_text.tStartRefresh = tThisFlipGlobal  # on global time
            # win.timeOnFlip(instrucciones_text, 'tStartRefresh')  # time at next scr refresh
            # update status
            instrucciones_text.status = STARTED
            instrucciones_text.setAutoDraw(True)
        
        # if instrucciones_text is active this frame...
        if instrucciones_text.status == STARTED:
            # update params
            pass
        
        # *key_instrucciones* updates
        waitOnFlip = False
        
        # if key_instrucciones is starting this frame...
        if key_instrucciones.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_instrucciones.frameNStart = frameN  # exact frame index
            key_instrucciones.tStart = t  # local t and not account for scr refresh
            key_instrucciones.tStartRefresh = tThisFlipGlobal  # on global time
            # win.timeOnFlip(key_instrucciones, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            # thisExp.timestampOnFlip(win, 'key_instrucciones.started')
            # update status
            key_instrucciones.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instrucciones.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instrucciones.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instrucciones.status == STARTED and not waitOnFlip:
            theseKeys = key_instrucciones.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_instrucciones_allKeys.extend(theseKeys)
            if len(_key_instrucciones_allKeys):
                key_instrucciones.keys = _key_instrucciones_allKeys[-1].name  # just the last key pressed
                key_instrucciones.rt = _key_instrucciones_allKeys[-1].rt
                key_instrucciones.duration = _key_instrucciones_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruccionesComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instrucciones" ---
    for thisComponent in instruccionesComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # check responses
    if key_instrucciones.keys in ['', [], None]:  # No response was made
        key_instrucciones.keys = None
    thisExp.addData('key_instrucciones.keys',key_instrucciones.keys)
    if key_instrucciones.keys != None:  # we had a response
        thisExp.addData('key_instrucciones.rt', key_instrucciones.rt)
        thisExp.addData('key_instrucciones.duration', key_instrucciones.duration)
    thisExp.nextEntry()
    # the Routine "instrucciones" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    categorias = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('resources/categorias.xlsx'),
        seed=None, name='categorias')
    thisExp.addLoop(categorias)  # add the loop to the experiment
    thisCategoria = categorias.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisCategoria.rgb)
    if thisCategoria != None:
        for paramName in thisCategoria:
            globals()[paramName] = thisCategoria[paramName]
    
    for thisCategoria in categorias:
        currentLoop = categorias
        # thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisCategoria.rgb)
        if thisCategoria != None:
            for paramName in thisCategoria:
                globals()[paramName] = thisCategoria[paramName]
        
        # --- Prepare to start Routine "instrucciones_fijas2" ---
        continueRoutine = True
        # update component parameters for each repeat
        # skip this Routine if its 'Skip if' condition is True
        continueRoutine = continueRoutine and not (categorias.thisTrialN == 0)
        key_categorias.keys = []
        key_categorias.rt = []
        _key_categorias_allKeys = []
        # keep track of which components have finished
        instrucciones_fijas2Components = [text_6, key_categorias]
        for thisComponent in instrucciones_fijas2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "instrucciones_fijas2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_6* updates
            
            # if text_6 is starting this frame...
            if text_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_6.frameNStart = frameN  # exact frame index
                text_6.tStart = t  # local t and not account for scr refresh
                text_6.tStartRefresh = tThisFlipGlobal  # on global time
                # win.timeOnFlip(text_6, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_6.status = STARTED
                text_6.setAutoDraw(True)
            
            # if text_6 is active this frame...
            if text_6.status == STARTED:
                # update params
                pass
            
            # *key_categorias* updates
            waitOnFlip = False
            
            # if key_categorias is starting this frame...
            if key_categorias.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_categorias.frameNStart = frameN  # exact frame index
                key_categorias.tStart = t  # local t and not account for scr refresh
                key_categorias.tStartRefresh = tThisFlipGlobal  # on global time
                # win.timeOnFlip(key_categorias, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                # thisExp.timestampOnFlip(win, 'key_categorias.started')
                # update status
                key_categorias.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_categorias.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_categorias.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_categorias.status == STARTED and not waitOnFlip:
                theseKeys = key_categorias.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_categorias_allKeys.extend(theseKeys)
                if len(_key_categorias_allKeys):
                    key_categorias.keys = _key_categorias_allKeys[-1].name  # just the last key pressed
                    key_categorias.rt = _key_categorias_allKeys[-1].rt
                    key_categorias.duration = _key_categorias_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in instrucciones_fijas2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "instrucciones_fijas2" ---
        for thisComponent in instrucciones_fijas2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # check responses
        if key_categorias.keys in ['', [], None]:  # No response was made
            key_categorias.keys = None
        categorias.addData('key_categorias.keys',key_categorias.keys)
        if key_categorias.keys != None:  # we had a response
            categorias.addData('key_categorias.rt', key_categorias.rt)
            categorias.addData('key_categorias.duration', key_categorias.duration)
        # the Routine "instrucciones_fijas2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "cruz_fijacion" ---
        continueRoutine = True
        # update component parameters for each repeat
        # skip this Routine if its 'Skip if' condition is True
        continueRoutine = continueRoutine and not (key_insultos.keys == 's')
        # keep track of which components have finished
        cruz_fijacionComponents = [cruz_fijacion_text, blank]
        for thisComponent in cruz_fijacionComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "cruz_fijacion" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.65:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > 0.65-frameTolerance:
                continueRoutine = False
            
            # *cruz_fijacion_text* updates
            
            # if cruz_fijacion_text is starting this frame...
            if cruz_fijacion_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cruz_fijacion_text.frameNStart = frameN  # exact frame index
                cruz_fijacion_text.tStart = t  # local t and not account for scr refresh
                cruz_fijacion_text.tStartRefresh = tThisFlipGlobal  # on global time
                # win.timeOnFlip(cruz_fijacion_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                # thisExp.timestampOnFlip(win, 'cruz_fijacion_text.started')
                # update status
                cruz_fijacion_text.status = STARTED
                cruz_fijacion_text.setAutoDraw(True)
                thisExp.addData('inicio_cruz_fijacion', globalClock.getTime(format='%H:%M:%S.%f%z'))

            
            # if cruz_fijacion_text is active this frame...
            if cruz_fijacion_text.status == STARTED:
                # update params
                pass
            
            # if cruz_fijacion_text is stopping this frame...
            if cruz_fijacion_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cruz_fijacion_text.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    cruz_fijacion_text.tStop = t  # not accounting for scr refresh
                    cruz_fijacion_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    # thisExp.timestampOnFlip(win, 'cruz_fijacion_text.stopped')
                    # update status
                    cruz_fijacion_text.status = FINISHED
                    cruz_fijacion_text.setAutoDraw(False)
                    thisExp.addData('fin_cruz_fijacion', globalClock.getTime(format='%H:%M:%S.%f%z'))

            
            # *blank* updates
            
            # if blank is starting this frame...
            if blank.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                blank.frameNStart = frameN  # exact frame index
                blank.tStart = t  # local t and not account for scr refresh
                blank.tStartRefresh = tThisFlipGlobal  # on global time
                # win.timeOnFlip(blank, 'tStartRefresh')  # time at next scr refresh
                # update status
                blank.status = STARTED
                blank.setAutoDraw(True)
            
            # if blank is active this frame...
            if blank.status == STARTED:
                # update params
                pass
            
            # if blank is stopping this frame...
            if blank.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > blank.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    blank.tStop = t  # not accounting for scr refresh
                    blank.frameNStop = frameN  # exact frame index
                    # update status
                    blank.status = FINISHED
                    blank.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in cruz_fijacionComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "cruz_fijacion" ---
        for thisComponent in cruz_fijacionComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.650000)
        
        # --- Prepare to start Routine "grabacion_con_estimulo" ---
        continueRoutine = True
        # update component parameters for each repeat
        # skip this Routine if its 'Skip if' condition is True
        continueRoutine = continueRoutine and not (key_insultos.keys == 's')
        # Run 'Begin Routine' code from code
        recorder.start_recording(f'data/{expInfo["Nombre"]}/HospitalItaliano_{expInfo["Nombre"]}_{estimulo}.wav')
        thisExp.addData('inicio_grabacion', globalClock.getTime(format='%H:%M:%S.%f%z'))

        text_estimulo.setText(estimulo)
        sound_1.setSound('resources/beep.wav', secs=0.5, hamming=True)
        sound_1.setVolume(1.0, log=False)
        sound_1.seek(0)
        interrupt.keys = []
        interrupt.rt = []
        _interrupt_allKeys = []
        # keep track of which components have finished
        grabacion_con_estimuloComponents = [text_estimulo, sound_1, blank_2, interrupt]
        for thisComponent in grabacion_con_estimuloComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "grabacion_con_estimulo" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 65.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_estimulo* updates
            
            # if text_estimulo is starting this frame...
            if text_estimulo.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                text_estimulo.frameNStart = frameN  # exact frame index
                text_estimulo.tStart = t  # local t and not account for scr refresh
                text_estimulo.tStartRefresh = tThisFlipGlobal  # on global time
                # win.timeOnFlip(text_estimulo, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                # thisExp.timestampOnFlip(win, 'text_estimulo.started')
                # update status
                text_estimulo.status = STARTED
                text_estimulo.setAutoDraw(True)
                thisExp.addData('inicio_estimulo', globalClock.getTime(format='%H:%M:%S.%f%z'))

            
            # if text_estimulo is active this frame...
            if text_estimulo.status == STARTED:
                # update params
                pass
            
            # if text_estimulo is stopping this frame...
            if text_estimulo.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_estimulo.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    text_estimulo.tStop = t  # not accounting for scr refresh
                    text_estimulo.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    # thisExp.timestampOnFlip(win, 'text_estimulo.stopped')
                    # update status
                    text_estimulo.status = FINISHED
                    text_estimulo.setAutoDraw(False)
                    thisExp.addData('fin_estimulo', globalClock.getTime(format='%H:%M:%S.%f%z'))

            
            # if sound_1 is starting this frame...
            if sound_1.status == NOT_STARTED and tThisFlip >= 4.5-frameTolerance:
                # keep track of start time/frame for later
                sound_1.frameNStart = frameN  # exact frame index
                sound_1.tStart = t  # local t and not account for scr refresh
                sound_1.tStartRefresh = tThisFlipGlobal  # on global time
                # update status
                sound_1.status = STARTED
                sound_1.play(when=win)  # sync with win flip
                thisExp.addData('inicio_beep', globalClock.getTime(format='%H:%M:%S.%f%z'))

            
            # if sound_1 is stopping this frame...
            if sound_1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_1.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_1.tStop = t  # not accounting for scr refresh
                    sound_1.frameNStop = frameN  # exact frame index
                    # update status
                    sound_1.status = FINISHED
                    sound_1.stop()
                    thisExp.addData('fin_beep', globalClock.getTime(format='%H:%M:%S.%f%z'))

            # update sound_1 status according to whether it's playing
            if sound_1.isPlaying:
                sound_1.status = STARTED
            elif sound_1.isFinished:
                sound_1.status = FINISHED
            
            # *blank_2* updates
            
            # if blank_2 is starting this frame...
            if blank_2.status == NOT_STARTED and tThisFlip >= 5-frameTolerance:
                # keep track of start time/frame for later
                blank_2.frameNStart = frameN  # exact frame index
                blank_2.tStart = t  # local t and not account for scr refresh
                blank_2.tStartRefresh = tThisFlipGlobal  # on global time
                # win.timeOnFlip(blank_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                blank_2.status = STARTED
                blank_2.setAutoDraw(True)
            
            # if blank_2 is active this frame...
            if blank_2.status == STARTED:
                # update params
                pass
            
            # if blank_2 is stopping this frame...
            if blank_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > blank_2.tStartRefresh + 60-frameTolerance:
                    # keep track of stop time/frame for later
                    blank_2.tStop = t  # not accounting for scr refresh
                    blank_2.frameNStop = frameN  # exact frame index
                    # update status
                    blank_2.status = FINISHED
                    blank_2.setAutoDraw(False)
            
            # *interrupt* updates
            waitOnFlip = False
            
            # if interrupt is starting this frame...
            if interrupt.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
                # keep track of start time/frame for later
                interrupt.frameNStart = frameN  # exact frame index
                interrupt.tStart = t  # local t and not account for scr refresh
                interrupt.tStartRefresh = tThisFlipGlobal  # on global time
                # win.timeOnFlip(interrupt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                # thisExp.timestampOnFlip(win, 'interrupt.started')
                # update status
                interrupt.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(interrupt.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(interrupt.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if interrupt is stopping this frame...
            if interrupt.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > interrupt.tStartRefresh + 60-frameTolerance:
                    # keep track of stop time/frame for later
                    interrupt.tStop = t  # not accounting for scr refresh
                    interrupt.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    # thisExp.timestampOnFlip(win, 'interrupt.stopped')
                    # update status
                    interrupt.status = FINISHED
                    interrupt.status = FINISHED
            if interrupt.status == STARTED and not waitOnFlip:
                theseKeys = interrupt.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _interrupt_allKeys.extend(theseKeys)
                if len(_interrupt_allKeys):
                    interrupt.keys = _interrupt_allKeys[-1].name  # just the last key pressed
                    interrupt.rt = _interrupt_allKeys[-1].rt
                    interrupt.duration = _interrupt_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in grabacion_con_estimuloComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "grabacion_con_estimulo" ---
        for thisComponent in grabacion_con_estimuloComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
 
        
        # Run 'End Routine' code from code
        recorder.stop_recording()
        thisExp.addData('fin_grabacion', globalClock.getTime(format='%H:%M:%S.%f%z'))
        # thisExp.addData('fin_grabacion', globalClock.getTime(format='%H:%M:%S.%f%z'))
        sound_1.pause()  # ensure sound has stopped at end of Routine
        # check responses
        if interrupt.keys in ['', [], None]:  # No response was made
            interrupt.keys = None
        categorias.addData('interrupt.keys',interrupt.keys)
        if interrupt.keys != None:  # we had a response
            categorias.addData('interrupt.rt', interrupt.rt)
            categorias.addData('interrupt.duration', interrupt.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-65.000000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'categorias'
    
    
    # --- Prepare to start Routine "Instrucciones_Opcional" ---
    continueRoutine = True
    # update component parameters for each repeat
    key_insultos.keys = []
    key_insultos.rt = []
    _key_insultos_allKeys = []
    # keep track of which components have finished
    Instrucciones_OpcionalComponents = [text_10, key_insultos]
    for thisComponent in Instrucciones_OpcionalComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Instrucciones_Opcional" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_10* updates
        
        # if text_10 is starting this frame...
        if text_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_10.frameNStart = frameN  # exact frame index
            text_10.tStart = t  # local t and not account for scr refresh
            text_10.tStartRefresh = tThisFlipGlobal  # on global time
            # win.timeOnFlip(text_10, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_10.status = STARTED
            text_10.setAutoDraw(True)
        
        # if text_10 is active this frame...
        if text_10.status == STARTED:
            # update params
            pass
        
        # *key_insultos* updates
        waitOnFlip = False
        
        # if key_insultos is starting this frame...
        if key_insultos.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_insultos.frameNStart = frameN  # exact frame index
            key_insultos.tStart = t  # local t and not account for scr refresh
            key_insultos.tStartRefresh = tThisFlipGlobal  # on global time
            # win.timeOnFlip(key_insultos, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            # thisExp.timestampOnFlip(win, 'key_insultos.started')
            # update status
            key_insultos.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_insultos.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_insultos.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_insultos.status == STARTED and not waitOnFlip:
            theseKeys = key_insultos.getKeys(keyList=['space','s'], ignoreKeys=["escape"], waitRelease=False)
            _key_insultos_allKeys.extend(theseKeys)
            if len(_key_insultos_allKeys):
                key_insultos.keys = _key_insultos_allKeys[-1].name  # just the last key pressed
                key_insultos.rt = _key_insultos_allKeys[-1].rt
                key_insultos.duration = _key_insultos_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Instrucciones_OpcionalComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instrucciones_Opcional" ---
    for thisComponent in Instrucciones_OpcionalComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # check responses
    if key_insultos.keys in ['', [], None]:  # No response was made
        key_insultos.keys = None
    thisExp.addData('key_insultos.keys',key_insultos.keys)
    if key_insultos.keys != None:  # we had a response
        thisExp.addData('key_insultos.rt', key_insultos.rt)
        thisExp.addData('key_insultos.duration', key_insultos.duration)
    thisExp.nextEntry()
    # the Routine "Instrucciones_Opcional" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    insultos = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('resources/opcional.xlsx', selection='0'),
        seed=None, name='insultos')
    thisExp.addLoop(insultos)  # add the loop to the experiment
    thisInsulto = insultos.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisInsulto.rgb)
    if thisInsulto != None:
        for paramName in thisInsulto:
            globals()[paramName] = thisInsulto[paramName]
    
    for thisInsulto in insultos:
        currentLoop = insultos
        # thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisInsulto.rgb)
        if thisInsulto != None:
            for paramName in thisInsulto:
                globals()[paramName] = thisInsulto[paramName]
        
        # --- Prepare to start Routine "cruz_fijacion" ---
        continueRoutine = True
        # update component parameters for each repeat
        # skip this Routine if its 'Skip if' condition is True
        continueRoutine = continueRoutine and not (key_insultos.keys == 's')
        # keep track of which components have finished
        cruz_fijacionComponents = [cruz_fijacion_text, blank]
        for thisComponent in cruz_fijacionComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "cruz_fijacion" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.65:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > 0.65-frameTolerance:
                continueRoutine = False
            
            # *cruz_fijacion_text* updates
            
            # if cruz_fijacion_text is starting this frame...
            if cruz_fijacion_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cruz_fijacion_text.frameNStart = frameN  # exact frame index
                cruz_fijacion_text.tStart = t  # local t and not account for scr refresh
                cruz_fijacion_text.tStartRefresh = tThisFlipGlobal  # on global time
                # win.timeOnFlip(cruz_fijacion_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                # thisExp.timestampOnFlip(win, 'cruz_fijacion_text.started')
                # update status
                cruz_fijacion_text.status = STARTED
                cruz_fijacion_text.setAutoDraw(True)
                thisExp.addData('inicio_cruz_fijacion', globalClock.getTime(format='%H:%M:%S.%f%z'))

            
            # if cruz_fijacion_text is active this frame...
            if cruz_fijacion_text.status == STARTED:
                # update params
                pass
            
            # if cruz_fijacion_text is stopping this frame...
            if cruz_fijacion_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cruz_fijacion_text.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    cruz_fijacion_text.tStop = t  # not accounting for scr refresh
                    cruz_fijacion_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    # thisExp.timestampOnFlip(win, 'cruz_fijacion_text.stopped')
                    # update status
                    cruz_fijacion_text.status = FINISHED
                    cruz_fijacion_text.setAutoDraw(False)
                    thisExp.addData('fin_cruz_fijacion', globalClock.getTime(format='%H:%M:%S.%f%z'))

            
            # *blank* updates
            
            # if blank is starting this frame...
            if blank.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                blank.frameNStart = frameN  # exact frame index
                blank.tStart = t  # local t and not account for scr refresh
                blank.tStartRefresh = tThisFlipGlobal  # on global time
                # win.timeOnFlip(blank, 'tStartRefresh')  # time at next scr refresh
                # update status
                blank.status = STARTED
                blank.setAutoDraw(True)
            
            # if blank is active this frame...
            if blank.status == STARTED:
                # update params
                pass
            
            # if blank is stopping this frame...
            if blank.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > blank.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    blank.tStop = t  # not accounting for scr refresh
                    blank.frameNStop = frameN  # exact frame index
                    # update status
                    blank.status = FINISHED
                    blank.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in cruz_fijacionComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "cruz_fijacion" ---
        for thisComponent in cruz_fijacionComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.650000)
        
        # --- Prepare to start Routine "grabacion_con_estimulo" ---
        continueRoutine = True
        # update component parameters for each repeat
        # thisExp.addData('grabacion_con_estimulo.started', globalClock.getTime(format='%H:%M:%S.%f%z'))
        
        # skip this Routine if its 'Skip if' condition is True
        continueRoutine = continueRoutine and not (key_insultos.keys == 's')
        # Run 'Begin Routine' code from code
        recorder.start_recording(f'data/{expInfo["Nombre"]}/HospitalItaliano_{expInfo["Nombre"]}_{estimulo}.wav')
        thisExp.addData('inicio_grabacion', globalClock.getTime(format='%H:%M:%S.%f%z'))

        text_estimulo.setText(estimulo)
        sound_1.setSound('resources/beep.wav', secs=0.5, hamming=True)
        sound_1.setVolume(1.0, log=False)
        sound_1.seek(0)
        interrupt.keys = []
        interrupt.rt = []
        _interrupt_allKeys = []
        # keep track of which components have finished
        grabacion_con_estimuloComponents = [text_estimulo, sound_1, blank_2, interrupt]
        for thisComponent in grabacion_con_estimuloComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "grabacion_con_estimulo" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 65.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_estimulo* updates
            
            # if text_estimulo is starting this frame...
            if text_estimulo.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                text_estimulo.frameNStart = frameN  # exact frame index
                text_estimulo.tStart = t  # local t and not account for scr refresh
                text_estimulo.tStartRefresh = tThisFlipGlobal  # on global time
                # win.timeOnFlip(text_estimulo, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                # thisExp.timestampOnFlip(win, 'text_estimulo.started')
                # update status
                text_estimulo.status = STARTED
                text_estimulo.setAutoDraw(True)
                thisExp.addData('inicio_estimulo', globalClock.getTime(format='%H:%M:%S.%f%z'))

            
            # if text_estimulo is active this frame...
            if text_estimulo.status == STARTED:
                # update params
                pass
            
            # if text_estimulo is stopping this frame...
            if text_estimulo.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_estimulo.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    text_estimulo.tStop = t  # not accounting for scr refresh
                    text_estimulo.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    # thisExp.timestampOnFlip(win, 'text_estimulo.stopped')
                    # update status
                    text_estimulo.status = FINISHED
                    text_estimulo.setAutoDraw(False)
                    thisExp.addData('fin_estimulo', globalClock.getTime(format='%H:%M:%S.%f%z'))

            
            # if sound_1 is starting this frame...
            if sound_1.status == NOT_STARTED and tThisFlip >= 4.5-frameTolerance:
                # keep track of start time/frame for later
                sound_1.frameNStart = frameN  # exact frame index
                sound_1.tStart = t  # local t and not account for scr refresh
                sound_1.tStartRefresh = tThisFlipGlobal  # on global time
                # update status
                sound_1.status = STARTED
                sound_1.play(when=win)  # sync with win flip
                thisExp.addData('inicio_beep', globalClock.getTime(format='%H:%M:%S.%f%z'))

            
            # if sound_1 is stopping this frame...
            if sound_1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_1.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_1.tStop = t  # not accounting for scr refresh
                    sound_1.frameNStop = frameN  # exact frame index
                    # update status
                    sound_1.status = FINISHED
                    sound_1.stop()
                    thisExp.addData('fin_beep', globalClock.getTime(format='%H:%M:%S.%f%z'))

            # update sound_1 status according to whether it's playing
            if sound_1.isPlaying:
                sound_1.status = STARTED
            elif sound_1.isFinished:
                sound_1.status = FINISHED
            
            # *blank_2* updates
            
            # if blank_2 is starting this frame...
            if blank_2.status == NOT_STARTED and tThisFlip >= 5-frameTolerance:
                # keep track of start time/frame for later
                blank_2.frameNStart = frameN  # exact frame index
                blank_2.tStart = t  # local t and not account for scr refresh
                blank_2.tStartRefresh = tThisFlipGlobal  # on global time
                # win.timeOnFlip(blank_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                blank_2.status = STARTED
                blank_2.setAutoDraw(True)
            
            # if blank_2 is active this frame...
            if blank_2.status == STARTED:
                # update params
                pass
            
            # if blank_2 is stopping this frame...
            if blank_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > blank_2.tStartRefresh + 60-frameTolerance:
                    # keep track of stop time/frame for later
                    blank_2.tStop = t  # not accounting for scr refresh
                    blank_2.frameNStop = frameN  # exact frame index
                    # update status
                    blank_2.status = FINISHED
                    blank_2.setAutoDraw(False)
            
            # *interrupt* updates
            waitOnFlip = False
            
            # if interrupt is starting this frame...
            if interrupt.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
                # keep track of start time/frame for later
                interrupt.frameNStart = frameN  # exact frame index
                interrupt.tStart = t  # local t and not account for scr refresh
                interrupt.tStartRefresh = tThisFlipGlobal  # on global time
                # win.timeOnFlip(interrupt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                # thisExp.timestampOnFlip(win, 'interrupt.started')
                # update status
                interrupt.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(interrupt.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(interrupt.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if interrupt is stopping this frame...
            if interrupt.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > interrupt.tStartRefresh + 60-frameTolerance:
                    # keep track of stop time/frame for later
                    interrupt.tStop = t  # not accounting for scr refresh
                    interrupt.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    # thisExp.timestampOnFlip(win, 'interrupt.stopped')
                    # update status
                    interrupt.status = FINISHED
                    interrupt.status = FINISHED
            if interrupt.status == STARTED and not waitOnFlip:
                theseKeys = interrupt.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _interrupt_allKeys.extend(theseKeys)
                if len(_interrupt_allKeys):
                    interrupt.keys = _interrupt_allKeys[-1].name  # just the last key pressed
                    interrupt.rt = _interrupt_allKeys[-1].rt
                    interrupt.duration = _interrupt_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in grabacion_con_estimuloComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "grabacion_con_estimulo" ---
        for thisComponent in grabacion_con_estimuloComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('grabacion_con_estimulo.stopped', globalClock.getTime(format='%H:%M:%S.%f%z'))
        # Run 'End Routine' code from code
        recorder.stop_recording()
        thisExp.addData('fin_grabacion', globalClock.getTime(format='%H:%M:%S.%f%z'))
        sound_1.pause()  # ensure sound has stopped at end of Routine
        # check responses
        if interrupt.keys in ['', [], None]:  # No response was made
            interrupt.keys = None
        insultos.addData('interrupt.keys',interrupt.keys)
        if interrupt.keys != None:  # we had a response
            insultos.addData('interrupt.rt', interrupt.rt)
            insultos.addData('interrupt.duration', interrupt.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-65.000000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'insultos'
    
    
    # --- Prepare to start Routine "instrucciones" ---
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_2
    cont_inst += 1
    key_insultos.keys = []
    instrucciones_text.setText(open(f'resources/TextosInstrucciones/{cont_inst}.txt', encoding='utf-8').read()
    )
    key_instrucciones.keys = []
    key_instrucciones.rt = []
    _key_instrucciones_allKeys = []
    # keep track of which components have finished
    instruccionesComponents = [instrucciones_text, key_instrucciones]
    for thisComponent in instruccionesComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instrucciones" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instrucciones_text* updates
        
        # if instrucciones_text is starting this frame...
        if instrucciones_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instrucciones_text.frameNStart = frameN  # exact frame index
            instrucciones_text.tStart = t  # local t and not account for scr refresh
            instrucciones_text.tStartRefresh = tThisFlipGlobal  # on global time
            # win.timeOnFlip(instrucciones_text, 'tStartRefresh')  # time at next scr refresh
            # update status
            instrucciones_text.status = STARTED
            instrucciones_text.setAutoDraw(True)
        
        # if instrucciones_text is active this frame...
        if instrucciones_text.status == STARTED:
            # update params
            pass
        
        # *key_instrucciones* updates
        waitOnFlip = False
        
        # if key_instrucciones is starting this frame...
        if key_instrucciones.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_instrucciones.frameNStart = frameN  # exact frame index
            key_instrucciones.tStart = t  # local t and not account for scr refresh
            key_instrucciones.tStartRefresh = tThisFlipGlobal  # on global time
            # win.timeOnFlip(key_instrucciones, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            # thisExp.timestampOnFlip(win, 'key_instrucciones.started')
            # update status
            key_instrucciones.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instrucciones.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instrucciones.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instrucciones.status == STARTED and not waitOnFlip:
            theseKeys = key_instrucciones.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_instrucciones_allKeys.extend(theseKeys)
            if len(_key_instrucciones_allKeys):
                key_instrucciones.keys = _key_instrucciones_allKeys[-1].name  # just the last key pressed
                key_instrucciones.rt = _key_instrucciones_allKeys[-1].rt
                key_instrucciones.duration = _key_instrucciones_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruccionesComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instrucciones" ---
    for thisComponent in instruccionesComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # check responses
    if key_instrucciones.keys in ['', [], None]:  # No response was made
        key_instrucciones.keys = None
    thisExp.addData('key_instrucciones.keys',key_instrucciones.keys)
    if key_instrucciones.keys != None:  # we had a response
        thisExp.addData('key_instrucciones.rt', key_instrucciones.rt)
        thisExp.addData('key_instrucciones.duration', key_instrucciones.duration)
    thisExp.nextEntry()
    # the Routine "instrucciones" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    letras = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('resources/letras.xlsx'),
        seed=None, name='letras')
    thisExp.addLoop(letras)  # add the loop to the experiment
    thisLetra = letras.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisLetra.rgb)
    if thisLetra != None:
        for paramName in thisLetra:
            globals()[paramName] = thisLetra[paramName]
    
    for thisLetra in letras:
        currentLoop = letras
        # thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisLetra.rgb)
        if thisLetra != None:
            for paramName in thisLetra:
                globals()[paramName] = thisLetra[paramName]
        
        # --- Prepare to start Routine "instruciones_fijas" ---
        continueRoutine = True
        # update component parameters for each repeat
        # skip this Routine if its 'Skip if' condition is True
        continueRoutine = continueRoutine and not (letras.thisTrialN == 0)
        key_letras.keys = []
        key_letras.rt = []
        _key_letras_allKeys = []
        # keep track of which components have finished
        instruciones_fijasComponents = [text_7, key_letras]
        for thisComponent in instruciones_fijasComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "instruciones_fijas" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_7* updates
            
            # if text_7 is starting this frame...
            if text_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_7.frameNStart = frameN  # exact frame index
                text_7.tStart = t  # local t and not account for scr refresh
                text_7.tStartRefresh = tThisFlipGlobal  # on global time
                # win.timeOnFlip(text_7, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_7.status = STARTED
                text_7.setAutoDraw(True)
            
            # if text_7 is active this frame...
            if text_7.status == STARTED:
                # update params
                pass
            
            # *key_letras* updates
            waitOnFlip = False
            
            # if key_letras is starting this frame...
            if key_letras.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_letras.frameNStart = frameN  # exact frame index
                key_letras.tStart = t  # local t and not account for scr refresh
                key_letras.tStartRefresh = tThisFlipGlobal  # on global time
                # win.timeOnFlip(key_letras, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                # thisExp.timestampOnFlip(win, 'key_letras.started')
                # update status
                key_letras.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_letras.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_letras.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_letras.status == STARTED and not waitOnFlip:
                theseKeys = key_letras.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_letras_allKeys.extend(theseKeys)
                if len(_key_letras_allKeys):
                    key_letras.keys = _key_letras_allKeys[-1].name  # just the last key pressed
                    key_letras.rt = _key_letras_allKeys[-1].rt
                    key_letras.duration = _key_letras_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in instruciones_fijasComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "instruciones_fijas" ---
        for thisComponent in instruciones_fijasComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # check responses
        if key_letras.keys in ['', [], None]:  # No response was made
            key_letras.keys = None
        letras.addData('key_letras.keys',key_letras.keys)
        if key_letras.keys != None:  # we had a response
            letras.addData('key_letras.rt', key_letras.rt)
            letras.addData('key_letras.duration', key_letras.duration)
        # the Routine "instruciones_fijas" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "cruz_fijacion" ---
        continueRoutine = True
        # update component parameters for each repeat
        # skip this Routine if its 'Skip if' condition is True
        continueRoutine = continueRoutine and not (key_insultos.keys == 's')
        # keep track of which components have finished
        cruz_fijacionComponents = [cruz_fijacion_text, blank]
        for thisComponent in cruz_fijacionComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "cruz_fijacion" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.65:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > 0.65-frameTolerance:
                continueRoutine = False
            
            # *cruz_fijacion_text* updates
            
            # if cruz_fijacion_text is starting this frame...
            if cruz_fijacion_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cruz_fijacion_text.frameNStart = frameN  # exact frame index
                cruz_fijacion_text.tStart = t  # local t and not account for scr refresh
                cruz_fijacion_text.tStartRefresh = tThisFlipGlobal  # on global time
                # win.timeOnFlip(cruz_fijacion_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                # thisExp.timestampOnFlip(win, 'cruz_fijacion_text.started')
                # update status
                cruz_fijacion_text.status = STARTED
                cruz_fijacion_text.setAutoDraw(True)
                thisExp.addData('inicio_cruz_fijacion', globalClock.getTime(format='%H:%M:%S.%f%z'))

            
            # if cruz_fijacion_text is active this frame...
            if cruz_fijacion_text.status == STARTED:
                # update params
                pass
            
            # if cruz_fijacion_text is stopping this frame...
            if cruz_fijacion_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cruz_fijacion_text.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    cruz_fijacion_text.tStop = t  # not accounting for scr refresh
                    cruz_fijacion_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    # thisExp.timestampOnFlip(win, 'cruz_fijacion_text.stopped')
                    # update status
                    cruz_fijacion_text.status = FINISHED
                    cruz_fijacion_text.setAutoDraw(False)
                    thisExp.addData('fin_cruz_fijacion', globalClock.getTime(format='%H:%M:%S.%f%z'))

            
            # *blank* updates
            
            # if blank is starting this frame...
            if blank.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                blank.frameNStart = frameN  # exact frame index
                blank.tStart = t  # local t and not account for scr refresh
                blank.tStartRefresh = tThisFlipGlobal  # on global time
                # win.timeOnFlip(blank, 'tStartRefresh')  # time at next scr refresh
                # update status
                blank.status = STARTED
                blank.setAutoDraw(True)
            
            # if blank is active this frame...
            if blank.status == STARTED:
                # update params
                pass
            
            # if blank is stopping this frame...
            if blank.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > blank.tStartRefresh + 0.15-frameTolerance:
                    # keep track of stop time/frame for later
                    blank.tStop = t  # not accounting for scr refresh
                    blank.frameNStop = frameN  # exact frame index
                    # update status
                    blank.status = FINISHED
                    blank.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in cruz_fijacionComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "cruz_fijacion" ---
        for thisComponent in cruz_fijacionComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.650000)
        
        # --- Prepare to start Routine "grabacion_con_estimulo" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('grabacion_con_estimulo.started', globalClock.getTime(format='%H:%M:%S.%f%z'))
        # skip this Routine if its 'Skip if' condition is True
        continueRoutine = continueRoutine and not (key_insultos.keys == 's')
        # Run 'Begin Routine' code from code
        recorder.start_recording(f'data/{expInfo["Nombre"]}/HospitalItaliano_{expInfo["Nombre"]}_{estimulo}.wav')
        thisExp.addData('inicio_grabacion', globalClock.getTime(format='%H:%M:%S.%f%z'))

        text_estimulo.setText(estimulo)
        sound_1.setSound('resources/beep.wav', secs=0.5, hamming=True)
        sound_1.setVolume(1.0, log=False)
        sound_1.seek(0)
        interrupt.keys = []
        interrupt.rt = []
        _interrupt_allKeys = []
        # keep track of which components have finished
        grabacion_con_estimuloComponents = [text_estimulo, sound_1, blank_2, interrupt]
        for thisComponent in grabacion_con_estimuloComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "grabacion_con_estimulo" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 65.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_estimulo* updates
            
            # if text_estimulo is starting this frame...
            if text_estimulo.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                text_estimulo.frameNStart = frameN  # exact frame index
                text_estimulo.tStart = t  # local t and not account for scr refresh
                text_estimulo.tStartRefresh = tThisFlipGlobal  # on global time
                # win.timeOnFlip(text_estimulo, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                # thisExp.timestampOnFlip(win, 'text_estimulo.started')
                # update status
                text_estimulo.status = STARTED
                text_estimulo.setAutoDraw(True)
                thisExp.addData('inicio_estimulo', globalClock.getTime(format='%H:%M:%S.%f%z'))

            
            # if text_estimulo is active this frame...
            if text_estimulo.status == STARTED:
                # update params
                pass
            
            # if text_estimulo is stopping this frame...
            if text_estimulo.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_estimulo.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    text_estimulo.tStop = t  # not accounting for scr refresh
                    text_estimulo.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    # thisExp.timestampOnFlip(win, 'text_estimulo.stopped')
                    # update status
                    text_estimulo.status = FINISHED
                    text_estimulo.setAutoDraw(False)
                    thisExp.addData('fin_estimulo', globalClock.getTime(format='%H:%M:%S.%f%z'))

            
            # if sound_1 is starting this frame...
            if sound_1.status == NOT_STARTED and tThisFlip >= 4.5-frameTolerance:
                # keep track of start time/frame for later
                sound_1.frameNStart = frameN  # exact frame index
                sound_1.tStart = t  # local t and not account for scr refresh
                sound_1.tStartRefresh = tThisFlipGlobal  # on global time
                # update status
                sound_1.status = STARTED
                sound_1.play(when=win)  # sync with win flip
                thisExp.addData('inicio_beep', globalClock.getTime(format='%H:%M:%S.%f%z'))
            
            # if sound_1 is stopping this frame...
            if sound_1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_1.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_1.tStop = t  # not accounting for scr refresh
                    sound_1.frameNStop = frameN  # exact frame index
                    # update status
                    sound_1.status = FINISHED
                    sound_1.stop()
                    thisExp.addData('fin_beep', globalClock.getTime(format='%H:%M:%S.%f%z'))

            # update sound_1 status according to whether it's playing
            if sound_1.isPlaying:
                sound_1.status = STARTED
            elif sound_1.isFinished:
                sound_1.status = FINISHED
            
            # *blank_2* updates
            
            # if blank_2 is starting this frame...
            if blank_2.status == NOT_STARTED and tThisFlip >= 5-frameTolerance:
                # keep track of start time/frame for later
                blank_2.frameNStart = frameN  # exact frame index
                blank_2.tStart = t  # local t and not account for scr refresh
                blank_2.tStartRefresh = tThisFlipGlobal  # on global time
                # win.timeOnFlip(blank_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                blank_2.status = STARTED
                blank_2.setAutoDraw(True)
            
            # if blank_2 is active this frame...
            if blank_2.status == STARTED:
                # update params
                pass
            
            # if blank_2 is stopping this frame...
            if blank_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > blank_2.tStartRefresh + 60-frameTolerance:
                    # keep track of stop time/frame for later
                    blank_2.tStop = t  # not accounting for scr refresh
                    blank_2.frameNStop = frameN  # exact frame index
                    # update status
                    blank_2.status = FINISHED
                    blank_2.setAutoDraw(False)
            
            # *interrupt* updates
            waitOnFlip = False
            
            # if interrupt is starting this frame...
            if interrupt.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
                # keep track of start time/frame for later
                interrupt.frameNStart = frameN  # exact frame index
                interrupt.tStart = t  # local t and not account for scr refresh
                interrupt.tStartRefresh = tThisFlipGlobal  # on global time
                # win.timeOnFlip(interrupt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                # thisExp.timestampOnFlip(win, 'interrupt.started')
                # update status
                interrupt.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(interrupt.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(interrupt.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if interrupt is stopping this frame...
            if interrupt.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > interrupt.tStartRefresh + 60-frameTolerance:
                    # keep track of stop time/frame for later
                    interrupt.tStop = t  # not accounting for scr refresh
                    interrupt.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    # thisExp.timestampOnFlip(win, 'interrupt.stopped')
                    # update status
                    interrupt.status = FINISHED
                    interrupt.status = FINISHED
            if interrupt.status == STARTED and not waitOnFlip:
                theseKeys = interrupt.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _interrupt_allKeys.extend(theseKeys)
                if len(_interrupt_allKeys):
                    interrupt.keys = _interrupt_allKeys[-1].name  # just the last key pressed
                    interrupt.rt = _interrupt_allKeys[-1].rt
                    interrupt.duration = _interrupt_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in grabacion_con_estimuloComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "grabacion_con_estimulo" ---
        for thisComponent in grabacion_con_estimuloComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('grabacion_con_estimulo.stopped', globalClock.getTime(format='%H:%M:%S.%f%z'))
        # Run 'End Routine' code from code
        recorder.stop_recording()
        thisExp.addData('fin_grabacion', globalClock.getTime(format='%H:%M:%S.%f%z'))

        sound_1.pause()  # ensure sound has stopped at end of Routine
        # check responses
        if interrupt.keys in ['', [], None]:  # No response was made
            interrupt.keys = None
        letras.addData('interrupt.keys',interrupt.keys)
        if interrupt.keys != None:  # we had a response
            letras.addData('interrupt.rt', interrupt.rt)
            letras.addData('interrupt.duration', interrupt.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-65.000000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'letras'
    
    
    # --- Prepare to start Routine "end" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('end.started', globalClock.getTime(format='%H:%M:%S.%f%z'))
    key_resp_4.keys = []
    key_resp_4.rt = []
    _key_resp_4_allKeys = []
    # keep track of which components have finished
    endComponents = [text_2, key_resp_4]
    for thisComponent in endComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_2* updates
        
        # if text_2 is starting this frame...
        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_2.frameNStart = frameN  # exact frame index
            text_2.tStart = t  # local t and not account for scr refresh
            text_2.tStartRefresh = tThisFlipGlobal  # on global time
            # win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_2.status = STARTED
            text_2.setAutoDraw(True)
        
        # if text_2 is active this frame...
        if text_2.status == STARTED:
            # update params
            pass
        
        # *key_resp_4* updates
        waitOnFlip = False
        
        # if key_resp_4 is starting this frame...
        if key_resp_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_4.frameNStart = frameN  # exact frame index
            key_resp_4.tStart = t  # local t and not account for scr refresh
            key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
            # win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            # thisExp.timestampOnFlip(win, 'key_resp_4.started')
            # update status
            key_resp_4.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_4.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_4.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_4_allKeys.extend(theseKeys)
            if len(_key_resp_4_allKeys):
                key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
                key_resp_4.rt = _key_resp_4_allKeys[-1].rt
                key_resp_4.duration = _key_resp_4_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in endComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end" ---
    for thisComponent in endComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('end.stopped', globalClock.getTime(format='%H:%M:%S.%f%z'))
    # check responses
    if key_resp_4.keys in ['', [], None]:  # No response was made
        key_resp_4.keys = None
    thisExp.addData('key_resp_4.keys',key_resp_4.keys)
    if key_resp_4.keys != None:  # we had a response
        thisExp.addData('key_resp_4.rt', key_resp_4.rt)
        thisExp.addData('key_resp_4.duration', key_resp_4.duration)
    thisExp.nextEntry()
    # the Routine "end" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='comma')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and # win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and # win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
