# bot.py
import os
import discord
import requests
import cv2
import numpy as np
import qr
import github
import io

intents = discord.Intents.default()
intents.typing = False
intents.presences = False
intents.message_content = True

discord_token = os.getenv('DISCORD_TOKEN')
gh_token = os.getenv('GITHUB_TOKEN')

gh_url = "https://github.com/SynthstromAudible/DelugeFirmware/commit/"

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')
    for guild in client.guilds:
        print(
            f'{client.user} is connected to the following guild:\n'
            f'{guild.name}(id: {guild.id})'
        )

@client.event
async def on_message(message):
    # Check if the message has any attachments
    if message.attachments:
        # Check if any attachment is an image
        for attachment in message.attachments:
            print(attachment.url.lower())
            if attachment.url.lower().split('?')[0].endswith(('png', 'jpeg', 'jpg', 'gif')):
                # Respond with a text message
                try:
                    code, overlay = qr.deluge_qr_url(attachment.url)
                except Exception:
                    code = []
                    overlay = None
                    # Swallow exceptions and simply don't respond - if the bot
                    # seemingly ignores an obvious image it can be debugged by
                    # giving the same photo to the recognizer offline.
                    pass
                if len(code) == 5:
                    st = ""
                    for fv in code[:4]:
                        st += f"0x{fv:08x}\n"
                    commit_fragment = f"{code[4]:04x}"
                    st += f"0x{commit_fragment}"

                    recent_commits = github.get_recent_commits(gh_token, per_page=30, max_commits=100)

                    matching_commits = [c for c in recent_commits if c.startswith(commit_fragment)]
                    _, img_encoded = cv2.imencode('.png', overlay)
                    img_bytes = img_encoded.tobytes()
                    if len(matching_commits) == 1:
                        ghmsg = gh_url+matching_commits[0]
                    else:
                        ghmsg = "I couldn't find a recent matching commit."
                        
                    await message.channel.send(content = f'Thanks for the image {message.author.mention}!, '+
                                               f'it decodes as:\n{st}\n'+ ghmsg,
                                               file = discord.File(io.BytesIO(img_bytes), 'deluge_qr_decode.png'))
                        
client.run(discord_token)
