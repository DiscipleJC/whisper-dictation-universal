-- Whisper Dictation: menu-bar indicator and notifications for Hammerspoon.
-- Copy to ~/.hammerspoon/init.lua (or require it from yours).
-- Reads the local daemon only (http://127.0.0.1:18765/ui) — nothing leaves this Mac.
-- Needs no Accessibility permission. Notifications must be allowed for Hammerspoon.

local BASE = "http://127.0.0.1:18765"
local POLL_SEC = 1

local bar = hs.menubar.new()
local last = nil                    -- previous /ui answer
local recWasHandsFree = false       -- the recording now running/just ended was hands-free
local recSec = 0                    -- its last known length
local textsAtRec = 0                -- daemon's text counter while it was still recording
local unreachable = 0

local function mmss(sec)
  sec = math.floor(sec or 0)
  return string.format("%d:%02d", sec // 60, sec % 60)
end

-- One line in ~/.hammerspoon/status.txt whenever what the icon shows changes:
-- a way to check the indicator without looking at the screen. No transcript text.
local lastNote = nil
local function noteStatus(line)
  if line:gsub("%d+:%d%d", "") == lastNote then return end   -- the running timer is not a change
  lastNote = line:gsub("%d+:%d%d", "")
  local f = io.open(os.getenv("HOME") .. "/.hammerspoon/status.txt", "a")
  if f then
    f:write(os.date("%Y-%m-%d %H:%M:%S") .. "  " .. line .. "\n")
    f:close()
  end
end

local function notify(text, onClick)
  noteStatus("notification: " .. text)
  local n = hs.notify.new(onClick, {
    title = "Whisper Dictation",
    informativeText = text,
    withdrawAfter = 0,              -- stays in Notification Centre
  })
  n:send()
end

local function meetingsDir()
  return (last and last.meetings_dir) or (os.getenv("HOME") .. "/Documents/whisper-meetings")
end

-- Newest file in the meetings folder; plain=true → *_plain.txt, else the by-speaker one.
local function newestMeeting(plain)
  local dir = meetingsDir()
  local best, bestName = nil, nil
  local ok, iter, obj = pcall(hs.fs.dir, dir)
  if not ok then return nil end
  for name in iter, obj do
    if name:match("^meeting_.*%.txt$") and ((name:match("_plain%.txt$") ~= nil) == plain) then
      if not bestName or name > bestName then bestName, best = name, dir .. "/" .. name end
    end
  end
  return best
end

local function copyFile(path)
  if not path then
    notify("No meeting file yet")
    return
  end
  local f = io.open(path, "r")
  if not f then
    notify("Cannot read " .. path)
    return
  end
  local text = f:read("a")
  f:close()
  hs.pasteboard.setContents(text)
  notify("Copied — press Cmd+V. " .. path:match("[^/]+$"))
end

local function post(path)
  hs.http.asyncPost(BASE .. path, "", nil, function() end)
end

local function title(s)
  if not s then return "🎙⚠️" end
  if s.state == "recording" then
    if s.hands_free then return "🔴 " .. mmss(s.recording_sec) end
    return "🔴"
  elseif s.state == "transcribing" then
    return "⏳"
  elseif s.meeting and s.meeting.splitting then
    return "🗣"
  end
  return "🎙"
end

local function statusLine(s)
  if not s then return "Daemon is not answering" end
  if s.state == "recording" then
    return (s.hands_free and "Recording (hands-free) " or "Recording ") .. mmss(s.recording_sec)
  elseif s.state == "transcribing" then
    return "Transcribing…"
  elseif s.meeting and s.meeting.splitting then
    return "Splitting the meeting by speaker…"
  end
  return "Idle — hold right Option: dictate into the window · double tap: meeting → clipboard + files"
end

bar:setMenu(function()
  local s = last
  local recording = s and s.state == "recording"
  return {
    { title = statusLine(s), disabled = true },
    { title = "-" },
    { title = recording and "Stop recording" or "Start hands-free recording",
      disabled = (s == nil) or (s.state == "transcribing"),
      fn = function() post("/toggle") end },
    { title = "-" },
    { title = "Copy last meeting — by speaker", fn = function() copyFile(newestMeeting(false)) end },
    { title = "Copy last meeting — plain text", fn = function() copyFile(newestMeeting(true)) end,
      disabled = newestMeeting(true) == nil },
    { title = "Open meetings folder", fn = function() hs.execute('/usr/bin/open "' .. meetingsDir() .. '"') end },
  }
end)

local function onState(s)
  local p = last
  if p then
    -- a recording became hands-free (double tap, chord or menu)
    if s.state == "recording" and s.hands_free and not (p.state == "recording" and p.hands_free) then
      notify("Recording started — tap right Option once to stop")
    end
    if s.state == "recording" then
      recWasHandsFree = s.hands_free or (p.state == "recording" and recWasHandsFree)
      recSec = s.recording_sec
      textsAtRec = s.texts or 0
    end
    if p.state == "recording" and s.state ~= "recording" and recWasHandsFree then
      notify("Recording stopped (" .. mmss(recSec) .. ") — transcribing")
    end
    if p.state == "transcribing" and s.state == "idle" and recWasHandsFree then
      if (s.texts or 0) <= textsAtRec then
        notify("Nothing was recognised in that recording")
      elseif s.meeting_mode and s.meeting and s.meeting.splitting then
        notify("Meeting text is on the clipboard — press Cmd+V. The by-speaker file will follow…")
      elseif s.meeting_mode then
        notify("Meeting text is on the clipboard — press Cmd+V. Also saved in the meetings folder")
      elseif s.meeting and s.meeting.splitting then
        notify("Text is ready (Cmd+V). Splitting by speaker in the background…")
      else
        notify("Text is ready")
      end
      recWasHandsFree = false
    end
    -- a by-speaker split has finished
    -- (seq only grows; after a daemon restart it starts from 0 again — not an event)
    if s.meeting and p.meeting and s.meeting.seq > p.meeting.seq then
      local file = s.meeting.last_file
      if s.meeting.last_failed then
        notify("Split by speaker failed — plain text kept" ..
               (file and (": " .. file:match("[^/]+$")) or ""))
      else
        notify("By speaker ready: " .. tostring(s.meeting.last_voices) .. " voices — click to copy. " ..
               (file and file:match("[^/]+$") or ""),
               function() copyFile(file) end)
      end
    end
  end
  last = s
  bar:setTitle(title(s))
  noteStatus(title(s) .. " | " .. statusLine(s))
end

local function poll()
  hs.http.asyncGet(BASE .. "/ui", nil, function(code, body)
    if code == 200 then
      local s = hs.json.decode(body)
      if s then
        unreachable = 0
        onState(s)
        return
      end
    end
    unreachable = unreachable + 1
    if unreachable >= 5 then        -- a daemon restart takes a few seconds; do not flicker
      last = nil
      bar:setTitle(title(nil))
    end
  end)
end

WhisperPollTimer = hs.timer.doEvery(POLL_SEC, poll)   -- global: must not be garbage-collected
WhisperBar = bar
poll()
