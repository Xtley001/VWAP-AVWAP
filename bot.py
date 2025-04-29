#!/usr/bin/env python3
import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import yaml
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters as fts
)

class VWAPTraderPro:
    def __init__(self):
        self.cfg = self._load_config()
        self.application = Application.builder().token(self.cfg['telegram']['token']).build()
        self.logger = self._setup_logging()
        
        # Trading state
        self.running = False
        self.mt5_connected = False
        self.daily_balance = 0
        self.active_positions = {}
        self.trading_disabled = False
        self.today_trades = 0
        self.valid_symbols = []
        
        # Register handlers
        self._register_handlers()

    def _load_config(self):
        """Load and validate configuration"""
        try:
            with open("config.yaml") as f:
                config = yaml.safe_load(f)
                
            # Validate required fields
            required = ['telegram.token', 'telegram.chat_id',
                       'mt5.account', 'mt5.password', 'mt5.server']
            for field in required:
                keys = field.split('.')
                temp = config
                for key in keys:
                    if key not in temp:
                        raise ValueError(f"Missing config: {field}")
                    temp = temp[key]
                    
            return config
        except Exception as e:
            print(f"Config error: {str(e)}")
            sys.exit(1)

    def _setup_logging(self):
        """Configure logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler("vwap_trader.log"),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('VWAPTraderPro')

    def _register_handlers(self):
        """Register Telegram command handlers"""
        handlers = [
            CommandHandler("start", self.start),
            CommandHandler("stop", self.stop),
            CommandHandler("status", self.status),
            CommandHandler("positions", self.get_positions),
            MessageHandler(fts.TEXT & ~fts.COMMAND, self.handle_message)
        ]
        for handler in handlers:
            self.application.add_handler(handler)
        self.application.add_error_handler(self._error_handler)

    async def _connect_mt5(self):
        """Connect to MT5 with any broker"""
        mt5.shutdown()  # Cleanup existing connection
        
        if not mt5.initialize():
            self.logger.error(f"MT5 init failed: {mt5.last_error()}")
            return False
            
        authorized = mt5.login(
            login=int(self.cfg['mt5']['account']),
            password=self.cfg['mt5']['password'],
            server=self.cfg['mt5']['server'],
            timeout=30000
        )
        
        if not authorized:
            self.logger.error(f"Login failed: {mt5.last_error()}")
            return False
            
        self.mt5_connected = True
        self.daily_balance = mt5.account_info().balance
        self.valid_symbols = self._get_valid_symbols()
        self.logger.info(f"Connected to {self.cfg['mt5']['server']}. Balance: {self.daily_balance}")
        return True

    def _get_valid_symbols(self):
        """Get tradable symbols from broker"""
        return [s.name for s in mt5.symbols_get() 
                if s.visible and s.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL
                and s.name in self.cfg['symbols']]

    def _check_market_hours(self):
        """Check if within trading hours"""
        now = datetime.now()
        return (now.weekday() in self.cfg['trading_hours']['trading_days'] and
                self.cfg['trading_hours']['start_hour'] <= now.hour < self.cfg['trading_hours']['end_hour'])

    def _check_drawdown(self):
        """Check daily drawdown limit"""
        equity = mt5.account_info().equity
        drawdown = ((self.daily_balance - equity) / self.daily_balance) * 100
        if drawdown >= self.cfg['risk']['max_drawdown_pct']:
            self.logger.error(f"Max drawdown reached: {drawdown:.2f}%")
            self.trading_disabled = True
            return False
        return True

    def _get_data(self, symbol, timeframe, bars=500):
        """Get OHLCV data"""
        tf_map = {
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1
        }
        rates = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, bars)
        if rates is None:
            self.logger.error(f"No data for {symbol} {timeframe}")
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def _calculate_indicators(self, df, symbol, timeframe):
        """Calculate all technical indicators"""
        # Price and Volume
        price_src = (df['high'] + df['low'] + df['close']) / 3
        df['vol_ma'] = df['real_volume'].rolling(20).mean()
        
        # VWAP
        df['cum_pv'] = (price_src * df['real_volume']).cumsum()
        df['cum_vol'] = df['real_volume'].cumsum()
        df['vwap'] = df['cum_pv'] / df['cum_vol']
        
        # AVWAP from swing points
        highs = (df['high'].rolling(5, center=True).max() == df['high']) & \
               (df['real_volume'] > df['vol_ma'] * self.cfg['avwap']['min_volume_ratio'])
        lows = (df['low'].rolling(5, center=True).min() == df['low']) & \
              (df['real_volume'] > df['vol_ma'] * self.cfg['avwap']['min_volume_ratio'])
        anchors = df[highs | lows]['time'].tolist()[-3:]  # Last 3 swing points
        
        for i, anchor in enumerate(anchors):
            mask = df['time'] >= anchor
            df.loc[mask, f'avwap_{i}'] = (
                price_src[mask] * df['real_volume'][mask]).cumsum() / \
                df['real_volume'][mask].cumsum()
        
        # Moving Averages
        for period in self.cfg['moving_averages'][timeframe]:
            df[f'ma_{period}'] = df['close'].rolling(period).mean()
        
        # ATR for risk management
        df['tr'] = np.maximum.reduce([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift()),
            abs(df['low'] - df['close'].shift())
        ])
        df['atr'] = df['tr'].rolling(14).mean()
        
        return df

    def _generate_trade_signal(self, df, symbol, timeframe):
        """Generate independent signals per timeframe"""
        last = df.iloc[-1]
        atr = last['atr']
        
        # Bullish Signal (independent checks)
        if (last['close'] > last['vwap'] and
            any(last['close'] > last[f'avwap_{i}'] for i in range(3) if f'avwap_{i}' in last and not pd.isna(last[f'avwap_{i}'])) and
            last['close'] > last[f'ma_{self.cfg["moving_averages"][timeframe][0]}'] and
            last['real_volume'] > last['vol_ma'] * 1.5):
            
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': mt5.ORDER_BUY,
                'entry': last['close'],
                'stop_loss': last['close'] - atr * 1.0,
                'take_profit': last['close'] + atr * self.cfg['risk']['rr_ratio'],
                'atr': atr
            }
        
        # Bearish Signal (independent checks)
        elif (last['close'] < last['vwap'] and
              any(last['close'] < last[f'avwap_{i}'] for i in range(3) if f'avwap_{i}' in last and not pd.isna(last[f'avwap_{i}'])) and
              last['close'] < last[f'ma_{self.cfg["moving_averages"][timeframe][0]}'] and
              last['real_volume'] > last['vol_ma'] * 1.5):
            
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': mt5.ORDER_SELL,
                'entry': last['close'],
                'stop_loss': last['close'] + atr * 1.0,
                'take_profit': last['close'] - atr * self.cfg['risk']['rr_ratio'],
                'atr': atr
            }
        
        return None

    async def _execute_trade(self, signal, chat_id):
        """Execute trade with risk management"""
        if (self.trading_disabled or 
            len(self.active_positions) >= self.cfg['risk']['max_trades'] or
            (self.cfg['risk']['single_pair'] and signal['symbol'] in self.active_positions)):
            return False
            
        # Calculate position size
        symbol_info = mt5.symbol_info(signal['symbol'])
        if not symbol_info:
            return False
            
        risk_amount = self.daily_balance * (self.cfg['risk']['per_trade_risk_pct'] / 100)
        points_diff = abs(signal['entry'] - signal['stop_loss'])
        lot_size = round(risk_amount / (points_diff * symbol_info.trade_tick_value), 2)
        lot_size = max(min(lot_size, symbol_info.volume_max), symbol_info.volume_min)
        
        # Prepare order
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": signal['symbol'],
            "volume": lot_size,
            "type": signal['direction'],
            "price": signal['entry'],
            "sl": signal['stop_loss'],
            "tp": signal['take_profit'],
            "deviation": 10,
            "magic": 123456,
            "comment": f"VWAP_{signal['timeframe']}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        # Send order
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Trade failed: {result.comment}")
            return False
            
        # Store position
        self.active_positions[signal['symbol']] = {
            'ticket': result.order,
            'timeframe': signal['timeframe'],
            'direction': "BUY" if signal['direction'] == mt5.ORDER_BUY else "SELL",
            'entry': signal['entry'],
            'sl': signal['stop_loss'],
            'tp': signal['take_profit'],
            'lot_size': lot_size,
            'atr': signal['atr']
        }
        
        self.today_trades += 1
        
        # Send alert
        await self._send_trade_alert(chat_id, signal, result, lot_size)
        return True

    async def _send_trade_alert(self, chat_id, signal, result, lot_size):
        """Send trade alert to Telegram"""
        direction = "BUY" if signal['direction'] == mt5.ORDER_BUY else "SELL"
        risk_amount = self.daily_balance * (self.cfg['risk']['per_trade_risk_pct'] / 100)
        
        message = (
            f"⚡ *Trade Executed* ({result.order})\n\n"
            f"• Symbol: `{signal['symbol']}` {signal['timeframe']}\n"
            f"• Direction: `{direction}`\n"
            f"• Entry: `{signal['entry']:.5f}`\n"
            f"• Stop Loss: `{signal['stop_loss']:.5f}`\n"
            f"• Take Profit: `{signal['take_profit']:.5f}`\n"
            f"• Lot Size: `{lot_size:.2f}`\n"
            f"• Risk: `${risk_amount:.2f}` ({self.cfg['risk']['per_trade_risk_pct']}%)\n"
            f"• RR: 1:{self.cfg['risk']['rr_ratio']}\n"
            f"• ATR: `{signal['atr']:.5f}`"
        )
        
        await self.application.bot.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode="Markdown"
        )

    def _update_trailing_stops(self):
        """Update trailing stops for all positions"""
        for symbol, pos in list(self.active_positions.items()):
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                continue
                
            current_price = tick.ask if pos['direction'] == "BUY" else tick.bid
            trail_dist = pos['atr'] * (self.cfg['risk']['trailing_stop_pct'] / 100)
            
            if pos['direction'] == "BUY":
                new_sl = current_price - trail_dist
                if new_sl > pos['sl']:
                    self._modify_sl(pos['ticket'], new_sl)
            else:
                new_sl = current_price + trail_dist
                if new_sl < pos['sl']:
                    self._modify_sl(pos['ticket'], new_sl)

    def _modify_sl(self, ticket, new_sl):
        """Modify stop loss of position"""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": new_sl,
            "deviation": 10,
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            # Update local position record
            for sym, pos in self.active_positions.items():
                if pos['ticket'] == ticket:
                    pos['sl'] = new_sl
            return True
        return False

    async def _check_daily_reset(self):
        """Reset daily metrics at midnight"""
        if datetime.now().hour == 0 and datetime.now().minute == 0:
            account = mt5.account_info()
            if account:
                self.daily_balance = account.balance
                self.trading_disabled = False
                self.today_trades = 0
                self.logger.info(f"Daily reset. New balance: {self.daily_balance}")

    async def _market_analysis(self, context: ContextTypes.DEFAULT_TYPE):
        """Main trading loop"""
        if not self.running or not self.mt5_connected:
            return
            
        await self._check_daily_reset()
        
        if not self._check_market_hours() or not self._check_drawdown():
            return
            
        self._update_trailing_stops()
        
        # Process each symbol and timeframe independently
        for symbol in self.valid_symbols:
            for timeframe in self.cfg['timeframes']:
                try:
                    df = self._get_data(symbol, timeframe)
                    if df is None:
                        continue
                        
                    df = self._calculate_indicators(df, symbol, timeframe)
                    signal = self._generate_trade_signal(df, symbol, timeframe)
                    
                    if signal:
                        await self._execute_trade(signal, context.job.chat_id)
                        
                except Exception as e:
                    self.logger.error(f"Error in {symbol} {timeframe}: {str(e)}")

    # Telegram command handlers
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self.running:
            await update.message.reply_text("⚠️ Bot is already running!")
            return
            
        if not await self._connect_mt5():
            await update.message.reply_text("❌ Failed to connect to MT5!")
            return
            
        self.running = True
        context.job_queue.run_repeating(
            self._market_analysis,
            interval=300,
            first=10,
            chat_id=update.effective_chat.id,
            name="market_analysis"
        )
        
        await update.message.reply_text(
            f"🚀 *VWAP Trader Started*\n\n"
            f"• Broker: `{self.cfg['mt5']['server']}`\n"
            f"• Account: `{self.cfg['mt5']['account']}`\n"
            f"• Balance: `${self.daily_balance:.2f}`\n"
            f"• Timeframes: `{' '.join(self.cfg['timeframes'])}`\n"
            f"• Trading Hours: `{self.cfg['trading_hours']['start_hour']}-{self.cfg['trading_hours']['end_hour']} (Mon-Fri)`"
        )

    async def stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.running:
            await update.message.reply_text("⚠️ Bot isn't running!")
            return
            
        # Remove all jobs
        for job in context.job_queue.jobs():
            job.schedule_removal()
            
        mt5.shutdown()
        self.running = False
        await update.message.reply_text(
            f"🛑 *Bot Stopped*\n\n"
            f"• Active Trades: `{len(self.active_positions)}`\n"
            f"• Today's Trades: `{self.today_trades}`"
        )

    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        account = mt5.account_info()
        equity = account.equity if account else 0
        drawdown = ((self.daily_balance - equity) / self.daily_balance) * 100 if self.daily_balance > 0 else 0
        
        await update.message.reply_text(
            f"📊 *Bot Status*\n\n"
            f"• Running: `{'Yes' if self.running else 'No'}`\n"
            f"• MT5 Connected: `{'Yes' if self.mt5_connected else 'No'}`\n"
            f"• Equity: `${equity:.2f}`\n"
            f"• Drawdown: `{drawdown:.2f}%`\n"
            f"• Active Trades: `{len(self.active_positions)}/{self.cfg['risk']['max_trades']}`\n"
            f"• Today's Trades: `{self.today_trades}`",
            parse_mode="Markdown"
        )

    async def get_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.active_positions:
            await update.message.reply_text("No active positions")
            return
            
        msg = "📊 *Active Positions*\n\n"
        for sym, pos in self.active_positions.items():
            tick = mt5.symbol_info_tick(sym)
            price = tick.ask if pos['direction'] == "BUY" else tick.bid if tick else pos['entry']
            pnl = (price - pos['entry']) * pos['lot_size'] * (1 if pos['direction'] == "BUY" else -1)
            
            msg += (
                f"• *{sym}* {pos['direction']} ({pos['ticket']})\n"
                f"  Entry: `{pos['entry']:.5f}` | Price: `{price:.5f}`\n"
                f"  SL: `{pos['sl']:.5f}` | TP: `{pos['tp']:.5f}`\n"
                f"  PnL: `${pnl:.2f}` | Lots: `{pos['lot_size']:.2f}`\n\n"
            )
            
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "ℹ️ Available commands:\n"
            "/start - Start trading\n"
            "/stop - Stop trading\n"
            "/status - Show current status\n"
            "/positions - List active trades"
        )

    async def _error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        self.logger.error(f"Error: {context.error}", exc_info=True)

    def run(self):
        self.logger.info("Starting VWAP Trader Pro...")
        self.application.run_polling()

if __name__ == "__main__":
    bot = VWAPTraderPro()
    bot.run()