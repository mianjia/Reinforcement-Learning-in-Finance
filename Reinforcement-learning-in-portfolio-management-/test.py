{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "usage: ipykernel_launcher.py [-h] [--sum] N [N ...]\n",
      "ipykernel_launcher.py: error: argument N: invalid int value: '/run/user/1000/jupyter/kernel-8d01c3f8-095f-4e2d-bef1-f80ea87eed02.json'\n"
     ]
    },
    {
     "ename": "SystemExit",
     "evalue": "2",
     "output_type": "error",
     "traceback": [
      "An exception has occurred, use %tb to see the full traceback.\n",
      "\u001b[0;31mSystemExit\u001b[0m\u001b[0;31m:\u001b[0m 2\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/annguyen/anaconda3/lib/python3.7/site-packages/IPython/core/interactiveshell.py:3304: UserWarning: To exit: use 'exit', 'quit', or Ctrl-D.\n",
      "  warn(\"To exit: use 'exit', 'quit', or Ctrl-D.\", stacklevel=1)\n"
     ]
    }
   ],
   "source": [
    "import argparse\n",
    "\n",
    "parser = argparse.ArgumentParser(description='Process some integers.')\n",
    "parser.add_argument('integers', metavar='N', type=int, nargs='+',\n",
    "                    help='an integer for the accumulator')\n",
    "parser.add_argument('--sum', dest='accumulate', action='store_const',\n",
    "                    const=sum, default=max,\n",
    "                    help='sum the integers (default: find the max)')\n",
    "\n",
    "args = parser.parse_args()\n",
    "print(args.accumulate(args.integers))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
