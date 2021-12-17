# -*- coding: utf-8 -*-
import datetime

# log samples
apache_error = '[Sun Mar 01 06:47:23.708355 2020] [:error] [pid 23834] [client 192.168.10.18:57608] PHP Warning:  Declaration of Horde_Form_Type_pgp::init($gpg, $temp_dir = NULL, $rows = NULL, $cols = NULL) should be compatible with Horde_Form_Type_longtext::init($rows = 8, $cols = 80, $helper = Array) in /usr/share/php/Horde/Form/Type.php on line 878, referer: http://mail.insect.com/nag/'
apache_access='192.168.10.18 - - [04/Mar/2020:13:55:12 +0000] "GET /./admin/./browse.asp?FilePath=c:\\&Opt=2&level=0 HTTP/1.1" 302 713 "-" "Mozilla/5.00 (Nikto/2.1.5) (Evasions:2) (Test:000102)"'
auth = 'Feb 29 00:09:01 mail-2 CRON[27152]: pam_unix(cron:session): session opened for user "../ by (uid=0)'
messages = 'Feb 29 01:05:15 mail-2 HORDE: [horde] Login success for sabra to horde (192.168.10.18) [pid 27376 on line 163 of "/var/www/mail.insect.com/login.php"]'
fast = '02/29/2020-01:05:15.685652  [**] [1:2012887:3] ET POLICY Http Client Body contains pass= in cleartext [**] [Classification: Potential Corporate Privacy Violation] [Priority: 1] {TCP} 192.168.10.18:42066 -> 192.168.10.21:80'
user = 'Feb 29 01:05:15 mail-2 HORDE: [horde] Login success for sabra to horde (192.168.10.18) [pid 27376 on line 163 of "/var/www/mail.insect.com/login.php"]'


def error_extract(log):
    str = log[1:20] + log[27:32]
    # 有的是引号开头的
    dt = datetime.datetime.strptime(str, "%a %b %d %X %Y")
    return dt, log[34:]
def access_extract(log):
    start=log.index('[')+1
    str=log[start:start+20]
    dt=datetime.datetime.strptime(str,"%d/%b/%Y:%X")
    return dt,log[:start-1]+log[start+28:]

def auth_extract(log):
    str = log[0:15] + " 2020"
    dt = datetime.datetime.strptime(str, "%b %d %X %Y")
    return dt, log[16:]


def mess_extract(log):
    str = log[0:15] + " 2020"
    dt = datetime.datetime.strptime(str, "%b %d %X %Y")
    return dt, log[16:]


def fast_extract(log):
    str = log[:19]
    dt = datetime.datetime.strptime(str, "%m/%d/%Y-%X")
    return dt, log[28:]


def user_extract(log):
    str = log[:15] + " 2020"
    dt = datetime.datetime.strptime(str, "%b %d %X %Y")
    return dt, log[16:]


match = {'auth': auth_extract,
         'messages': mess_extract,
         'user': user_extract,
         'apache_error': error_extract,
         'apache_access': access_extract,
         'suricata_fast': fast_extract}
# print(access_extract(apache_access))