#!/bin/bash

# Function to pause and wait for user input
pause() {
    echo ""
    read -p "按回车键继续..."
}

# Function to identify device by unplugging
identify_device() {
    echo ""
    echo "------------------------------------------------"
    echo " 识别设备端口号"
    echo "------------------------------------------------"
    echo "1. 请确保您要识别的设备已插入。"
    read -p "插入设备后，请按回车键..."
    
    echo "正在记录当前设备列表..."
    lsusb > /tmp/lsusb_before.txt
    ls -1 /dev/ > /tmp/dev_before.txt
    
    echo "2. 请拔出设备。"
    read -p "拔出设备后，请按回车键..."
    
    lsusb > /tmp/lsusb_after.txt
    ls -1 /dev/ > /tmp/dev_after.txt
    
    # Compare files to find the missing line
    DIFF=$(diff /tmp/lsusb_before.txt /tmp/lsusb_after.txt | grep "^< " | cut -c 3-)
    
    # Check for removed device nodes
    DEV_DIFF=$(diff /tmp/dev_before.txt /tmp/dev_after.txt | grep "^< " | awk '{print $NF}' | sort | uniq)

    if [ -z "$DIFF" ]; then
        echo "未检测到 USB 设备移除（lsusb 无变化）！请重试。"
        rm /tmp/lsusb_before.txt /tmp/lsusb_after.txt /tmp/dev_before.txt /tmp/dev_after.txt
        return
    fi
    
    echo "------------------------------------------------"
    echo "检测到的设备："
    echo "$DIFF"
    
    if [ -n "$DEV_DIFF" ]; then
        echo "检测到的设备节点 (Device Node):"
        for node in $DEV_DIFF; do
             # Filter out common noise if necessary, but usually /dev/ changes are relevant
             if [[ "$node" == tty* || "$node" == video* || "$node" == sd* || "$node" == input* ]]; then
                 echo "/dev/$node"
             fi
        done
    else
        echo "未检测到明显的 /dev/ 节点变化（可能是非字符/块设备）。"
    fi
    echo "------------------------------------------------"
    
    # Extract VID and PID
    # Example line: Bus 001 Device 005: ID 1a2c:4d5e Chip Name...
    VID=$(echo "$DIFF" | awk '{print $6}' | cut -d: -f1)
    PID=$(echo "$DIFF" | awk '{print $6}' | cut -d: -f2)
    
    if [ -z "$VID" ] || [ -z "$PID" ]; then
        echo "无法解析 VendorID 和 ProductID。"
        rm /tmp/lsusb_before.txt /tmp/lsusb_after.txt
        return
    fi
    
    echo "厂商 ID (Vendor ID): $VID"
    echo "产品 ID (Product ID): $PID"
    
    # Get more details using udevadm
    # We need to find a device path that matches this VID:PID
    # Since the device is unplugged, we can't query it directly now.
    # But wait, to write a rule, we usually want to know the Serial Number too.
    # If the device is unplugged, we can't get the Serial Number easily unless we used udevadm monitor before.
    # Or we can ask the user to plug it back in.
    
    rm /tmp/lsusb_before.txt /tmp/lsusb_after.txt /tmp/dev_before.txt /tmp/dev_after.txt
}

# Function to create udev rule
create_udev_rule() {
    echo ""
    echo "------------------------------------------------"
    echo " 创建 UDEV 规则"
    echo "------------------------------------------------"
    
    # 1. Ask for device node
    while true; do
        read -p "请输入端口号 (例如 /dev/ttyACM0): " DEV_NODE
        if [ -e "$DEV_NODE" ]; then
            echo "已找到设备: $DEV_NODE"
            break
        else
            echo "错误：未找到设备 $DEV_NODE，请检查拼写或设备是否已连接。"
            read -p "是否重试？(y/n): " retry
            if [[ "$retry" != "y" ]]; then
                return
            fi
        fi
    done
    
    # 2. Query serial number
    echo "正在查询设备信息..."
    SERIAL_INFO=$(udevadm info -a -n "$DEV_NODE" | grep "ATTRS{serial}")
    
    if [ -z "$SERIAL_INFO" ]; then
        echo "未找到该设备的序列号 (ATTRS{serial}) 信息。"
        echo "这可能是因为设备不支持序列号，或者权限不足。"
        return
    fi
    
    echo "查询到的序列号信息："
    echo "$SERIAL_INFO"
    echo "------------------------------------------------"
    
    # Extract the first serial (usually the device serial)
    # The output might have multiple lines, e.g. for parent hubs.
    # We typically want the first one that looks like a unique ID.
    # Let's list them and let user confirm or just pick the first one.
    
    # Simplified approach: extract the value inside quotes of the first match
    # ATTRS{serial}=="58FA083324" -> 58FA083324
    SERIAL_VAL=$(echo "$SERIAL_INFO" | head -n 1 | cut -d'"' -f2)
    
    echo "推荐使用的序列号: $SERIAL_VAL"
    
    # Check for duplicate serial in existing rules
    echo "正在检查现有规则中是否已包含此序列号..."
    
    # Check if directory exists and is not empty
    if [ -d "/etc/udev/rules.d/" ] && [ "$(ls -A /etc/udev/rules.d/)" ]; then
        EXISTING_RULES=$(grep -r "$SERIAL_VAL" /etc/udev/rules.d/ 2>/dev/null)
        
        if [ -n "$EXISTING_RULES" ]; then
            echo "警告！发现重复的序列号规则："
            echo "------------------------------------------------"
            
            # Format output for better readability
            # EXISTING_RULES contains lines like: /etc/udev/rules.d/xxx.rules:SUBSYSTEM=="tty", ...
            # We iterate line by line
            echo "$EXISTING_RULES" | while IFS= read -r line; do
                # Extract filename
                FILE=$(echo "$line" | cut -d: -f1)
                CONTENT=$(echo "$line" | cut -d: -f2-)
                
                # Extract attributes
                VID=$(echo "$CONTENT" | grep -o 'ATTRS{idVendor}=="[^"]*"' | cut -d'"' -f2)
                PID=$(echo "$CONTENT" | grep -o 'ATTRS{idProduct}=="[^"]*"' | cut -d'"' -f2)
                SERIAL=$(echo "$CONTENT" | grep -o 'ATTRS{serial}=="[^"]*"' | cut -d'"' -f2)
                SYMLINK=$(echo "$CONTENT" | grep -o 'SYMLINK+="[^"]*"' | cut -d'"' -f2)
                
                echo "文件路径: $FILE"
                echo "厂商 ID:  $VID"
                echo "产品 ID:  $PID"
                echo "序列号:   $SERIAL"
                echo "软链接:   $SYMLINK"
                echo "------------------------------------------------"
            done
            
            read -p "是否仍要继续？这可能导致冲突。(y/n): " confirm_dup
            if [[ "$confirm_dup" != "y" ]]; then
                echo "操作已取消。"
                return
            fi
        else
            echo "未发现重复规则，可以安全创建。"
        fi
    else
        echo "规则目录为空或不存在，无需检查重复。"
    fi
    
    read -p "是否使用此序列号创建规则？(y/n): " confirm
    if [[ "$confirm" != "y" ]]; then
        echo "操作已取消。"
        return
    fi
    
    # 3. Ask for filename and symlink
    read -p "请输入规则文件名 (例如 cobot_chassis.rules): " RULE_FILENAME
    # Add .rules extension if missing
    if [[ "$RULE_FILENAME" != *.rules ]]; then
        RULE_FILENAME="${RULE_FILENAME}.rules"
    fi
    
    read -p "请输入映射的链接名称 (例如 driver_chassis): " SYMLINK_NAME
    
    # 4. Construct the rule content
    # Note: We need VID and PID as well to be safe, but user asked to use Serial.
    # Usually VID/PID + Serial is best. Let's try to get VID/PID too.
    VID_INFO=$(udevadm info -a -n "$DEV_NODE" | grep "ATTRS{idVendor}" | head -n 1 | cut -d'"' -f2)
    PID_INFO=$(udevadm info -a -n "$DEV_NODE" | grep "ATTRS{idProduct}" | head -n 1 | cut -d'"' -f2)
    
    RULE_CONTENT="SUBSYSTEM==\"tty\", ATTRS{idVendor}==\"$VID_INFO\", ATTRS{idProduct}==\"$PID_INFO\", ATTRS{serial}==\"$SERIAL_VAL\", MODE=\"0666\", SYMLINK+=\"$SYMLINK_NAME\""
    
    echo "------------------------------------------------"
    echo "即将创建的文件: $RULE_FILENAME"
    echo "规则内容:"
    echo "$RULE_CONTENT"
    echo "------------------------------------------------"
    
    read -p "确认创建并应用吗？(y/n): " confirm_create
    if [[ "$confirm_create" != "y" ]]; then
        echo "操作已取消。"
        return
    fi
    
    # 5. Write to file and install
    # Write to a temp file first
    echo "$RULE_CONTENT" > "/tmp/$RULE_FILENAME"
    
    echo "正在安装规则..."
    sudo cp "/tmp/$RULE_FILENAME" "/etc/udev/rules.d/"
    sudo chmod 644 "/etc/udev/rules.d/$RULE_FILENAME"
    rm "/tmp/$RULE_FILENAME"
    
    echo "正在重新加载规则..."
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    
    echo "完成！"
    echo "请检查 /dev/$SYMLINK_NAME 是否存在："
    ls -l "/dev/$SYMLINK_NAME" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "成功：软链接已创建。"
    else
        echo "提示：如果未看到软链接，请尝试重新插拔设备。"
    fi
}

# Function to list existing Feite udev rules (Vendor ID 1a86)
list_feite_rules() {
    echo ""
    echo "------------------------------------------------"
    echo " 已配置的飞特 UDEV 规则 (Vendor ID: 1a86)"
    echo "------------------------------------------------"
    
    if [ ! -d "/etc/udev/rules.d/" ] || [ ! "$(ls -A /etc/udev/rules.d/)" ]; then
        echo "规则目录为空或不存在。"
        return
    fi
    
    # Grep for Vendor ID 1a86 in rules directory
    # 1a86 is QinHeng Electronics (often used by Feite servos/chassis)
    RULES=$(grep -r "ATTRS{idVendor}==\"1a86\"" /etc/udev/rules.d/ 2>/dev/null)
    
    if [ -z "$RULES" ]; then
        echo "未找到 Vendor ID 为 1a86 的相关规则。"
    else
        # Format and display
        echo "$RULES" | while IFS= read -r line; do
             # Extract filename
             FILE=$(echo "$line" | cut -d: -f1)
             CONTENT=$(echo "$line" | cut -d: -f2-)
             
             # Extract attributes
             PID=$(echo "$CONTENT" | grep -o 'ATTRS{idProduct}=="[^"]*"' | cut -d'"' -f2)
             SERIAL=$(echo "$CONTENT" | grep -o 'ATTRS{serial}=="[^"]*"' | cut -d'"' -f2)
             SYMLINK=$(echo "$CONTENT" | grep -o 'SYMLINK+="[^"]*"' | cut -d'"' -f2)
             
             echo "文件路径: $FILE"
             echo "产品 ID:  $PID"
             echo "序列号:   $SERIAL"
             echo "软链接:   $SYMLINK"
             echo "------------------------------------------------"
        done
    fi
}

# Main Menu
while true; do
    clear
    echo "========================================"
    echo "       Linux UDEV 规则配置工具    "
    echo "========================================"
    echo "1. 识别设备端口号"
    echo "2. 飞特舵机 udev 规则创建"
    echo "3. 查看已经配置的飞特 udev 规则"
    echo "4. 退出"
    echo "========================================"
    read -p "请输入您的选择 [1-4]: " choice
    
    case $choice in
        1)
            identify_device
            pause
            ;;
        2)
            create_udev_rule
            pause
            ;;
        3)
            list_feite_rules
            pause
            ;;
        4)
            echo "正在退出..."
            exit 0
            ;;
        *)
            echo "无效选项。请重试。"
            sleep 1
            ;;
    esac
done
